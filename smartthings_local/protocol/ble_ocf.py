"""Pure IoTivity BLE transport framing for OCF PDUs.

This module implements only the wire codec and reassembly state machine. It
does not open a Bluetooth connection or read or write a GATT characteristic.

The format follows IoTivity's ``cafragmentation.h`` and
``cafragmentation.c``:

* byte 0: start flag in bit 7, source port in bits 0 through 6;
* byte 1: secure flag in bit 7, destination port in bits 0 through 6;
* the first frame then carries the total PDU length as a four-byte,
  big-endian unsigned integer;
* continuation frames carry only the two-byte header before their payload.

IoTivity fills every non-final frame to the negotiated transport frame size.
The reassembler enforces that invariant, metadata consistency, and a
conservative PDU-size bound. The header has no sequence number, so no framing
codec can distinguish reordered or duplicated full-size continuation frames;
IoTivity relies on GATT's ordered transport for that property.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final

__all__ = [
    "BLE_FIRST_FRAME_OVERHEAD",
    "BLE_HEADER_SIZE",
    "BLE_LENGTH_HEADER_SIZE",
    "BLE_MAX_MTU",
    "BLE_MAX_PORT",
    "BLE_MIN_SOURCE_PORT",
    "BLE_MULTICAST_PORT",
    "BLE_WIRE_MAX_PDU_SIZE",
    "DEFAULT_MAX_PDU_SIZE",
    "AdaptiveBleOcfReassembler",
    "BleOcfCodecError",
    "BleOcfHeader",
    "BleOcfInterleavedFrameError",
    "BleOcfReassembler",
    "ReassembledBleOcfPdu",
    "decode_header",
    "encode_header",
    "fragment_pdu",
]

BLE_HEADER_SIZE: Final = 2
BLE_LENGTH_HEADER_SIZE: Final = 4
BLE_FIRST_FRAME_OVERHEAD: Final = BLE_HEADER_SIZE + BLE_LENGTH_HEADER_SIZE
BLE_MIN_SOURCE_PORT: Final = 1
BLE_MAX_PORT: Final = 127
BLE_MULTICAST_PORT: Final = 0
BLE_MAX_MTU: Final = 0xFFFF
BLE_WIRE_MAX_PDU_SIZE: Final = 0xFFFFFFFF

# A four-byte length field can describe almost 4 GiB, but accepting such a
# declaration by default would make a local peer an easy memory-exhaustion
# vector. OCF control PDUs are far smaller. Callers may select another bound,
# up to the on-wire uint32 maximum, when constructing the codec.
DEFAULT_MAX_PDU_SIZE: Final = 1024 * 1024


class BleOcfCodecError(ValueError):
    """An invalid or unsupported BLE OCF frame."""


class BleOcfInterleavedFrameError(BleOcfCodecError):
    """A frame belongs to a different PDU than the active reassembly."""


@dataclass(frozen=True, slots=True)
class BleOcfHeader:
    """Semantic representation of the two-byte IoTivity BLE header."""

    start: bool
    source_port: int
    secure: bool
    destination_port: int


@dataclass(frozen=True, slots=True, repr=False)
class ReassembledBleOcfPdu:
    """A complete OCF PDU and the transport metadata that carried it."""

    pdu: bytes
    source_port: int
    destination_port: int
    secure: bool

    def __repr__(self) -> str:
        """Return transport metadata without exposing the PDU bytes."""
        return (
            "ReassembledBleOcfPdu("
            f"pdu_length={len(self.pdu)}, source_port={self.source_port}, "
            f"destination_port={self.destination_port}, secure={self.secure})"
        )


def _byte_length(value: object, *, name: str) -> int:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    try:
        return value.nbytes if isinstance(value, memoryview) else len(value)
    except ValueError:
        raise TypeError(f"{name} must be an active bytes-like value") from None


def _coerce_bytes(
    value: object,
    *,
    name: str,
    max_length: int | None = None,
    limit_name: str = "maximum",
) -> bytes:
    length = _byte_length(value, name=name)
    if max_length is not None and length > max_length:
        raise BleOcfCodecError(
            f"{name} length {length} exceeds {limit_name} {max_length}"
        )
    try:
        return bytes(value)
    except ValueError:
        raise TypeError(f"{name} must be an active bytes-like value") from None


def _validate_port(port: object, *, source: bool) -> int:
    label = "source_port" if source else "destination_port"
    minimum = BLE_MIN_SOURCE_PORT if source else BLE_MULTICAST_PORT
    if isinstance(port, bool) or not isinstance(port, int):
        raise TypeError(f"{label} must be an integer")
    if not minimum <= port <= BLE_MAX_PORT:
        raise BleOcfCodecError(
            f"{label} must be in the range {minimum}..{BLE_MAX_PORT}"
        )
    return port


def _validate_flag(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _validate_mtu(mtu: object) -> int:
    if isinstance(mtu, bool) or not isinstance(mtu, int):
        raise TypeError("mtu must be an integer")
    # A non-empty PDU needs the two-byte header, four-byte length, and at
    # least one payload byte in its first frame.
    if not BLE_FIRST_FRAME_OVERHEAD < mtu <= BLE_MAX_MTU:
        raise BleOcfCodecError(
            f"mtu must be in the range {BLE_FIRST_FRAME_OVERHEAD + 1}..{BLE_MAX_MTU}"
        )
    return mtu


def _validate_max_pdu_size(max_pdu_size: object) -> int:
    if isinstance(max_pdu_size, bool) or not isinstance(max_pdu_size, int):
        raise TypeError("max_pdu_size must be an integer")
    if not 1 <= max_pdu_size <= BLE_WIRE_MAX_PDU_SIZE:
        raise BleOcfCodecError(
            f"max_pdu_size must be in the range 1..{BLE_WIRE_MAX_PDU_SIZE}"
        )
    return max_pdu_size


def encode_header(
    *,
    start: bool,
    source_port: int,
    secure: bool,
    destination_port: int,
) -> bytes:
    """Encode the two-byte IoTivity BLE transport header."""

    start = _validate_flag(start, name="start")
    secure = _validate_flag(secure, name="secure")
    source_port = _validate_port(source_port, source=True)
    destination_port = _validate_port(destination_port, source=False)
    return bytes(
        (
            (0x80 if start else 0) | source_port,
            (0x80 if secure else 0) | destination_port,
        )
    )


def decode_header(frame: bytes | bytearray | memoryview) -> BleOcfHeader:
    """Decode and validate the header at the beginning of ``frame``."""

    data = _coerce_bytes(frame, name="frame")
    if len(data) < BLE_HEADER_SIZE:
        raise BleOcfCodecError("BLE OCF frame is shorter than its two-byte header")

    first, second = data[:BLE_HEADER_SIZE]
    source_port = first & 0x7F
    if source_port < BLE_MIN_SOURCE_PORT:
        raise BleOcfCodecError("BLE OCF source port 0 is invalid")
    return BleOcfHeader(
        start=bool(first & 0x80),
        source_port=source_port,
        secure=bool(second & 0x80),
        destination_port=second & 0x7F,
    )


def fragment_pdu(
    pdu: bytes | bytearray | memoryview,
    *,
    mtu: int,
    source_port: int,
    destination_port: int,
    secure: bool,
    max_pdu_size: int = DEFAULT_MAX_PDU_SIZE,
) -> tuple[bytes, ...]:
    """Fragment one non-empty OCF PDU into IoTivity BLE frames.

    ``mtu`` is the maximum complete characteristic-value frame used by the
    IoTivity adapter. It is not the raw ATT MTU; a default ATT MTU of 23, for
    example, normally permits a 20-byte characteristic value.
    """

    mtu = _validate_mtu(mtu)
    max_pdu_size = _validate_max_pdu_size(max_pdu_size)
    source_port = _validate_port(source_port, source=True)
    destination_port = _validate_port(destination_port, source=False)
    secure = _validate_flag(secure, name="secure")
    data = _coerce_bytes(
        pdu,
        name="OCF PDU",
        max_length=max_pdu_size,
    )

    if not data:
        raise BleOcfCodecError("OCF PDU must not be empty")

    first_capacity = mtu - BLE_FIRST_FRAME_OVERHEAD
    continuation_capacity = mtu - BLE_HEADER_SIZE
    first_length = min(len(data), first_capacity)
    frames = [
        encode_header(
            start=True,
            source_port=source_port,
            secure=secure,
            destination_port=destination_port,
        )
        + struct.pack(">I", len(data))
        + data[:first_length]
    ]

    continuation_header = encode_header(
        start=False,
        source_port=source_port,
        secure=secure,
        destination_port=destination_port,
    )
    offset = first_length
    while offset < len(data):
        end = min(offset + continuation_capacity, len(data))
        frames.append(continuation_header + data[offset:end])
        offset = end
    return tuple(frames)


class BleOcfReassembler:
    """Strict, single-PDU IoTivity BLE reassembly state machine.

    Any malformed or interleaved frame aborts the active PDU before the error
    is raised. This fail-closed behavior prevents a later valid-looking tail
    from completing data that was already shown to be inconsistent.
    """

    def __init__(
        self,
        *,
        mtu: int,
        max_pdu_size: int = DEFAULT_MAX_PDU_SIZE,
    ) -> None:
        self._mtu = _validate_mtu(mtu)
        self._max_pdu_size = _validate_max_pdu_size(max_pdu_size)
        self.reset()

    @property
    def in_progress(self) -> bool:
        """Return whether a partial PDU is currently buffered."""

        return self._header is not None

    @property
    def buffered_bytes(self) -> int:
        """Return the number of PDU bytes retained for reassembly."""

        return len(self._buffer)

    def reset(self) -> None:
        """Discard any partial PDU."""

        self._header: BleOcfHeader | None = None
        self._expected_length = 0
        self._buffer = bytearray()

    def feed(
        self,
        frame: bytes | bytearray | memoryview,
    ) -> ReassembledBleOcfPdu | None:
        """Consume one complete GATT value and return a PDU when complete."""

        try:
            return self._feed(frame)
        except (BleOcfCodecError, TypeError):
            self.reset()
            raise

    def _feed(
        self,
        frame: bytes | bytearray | memoryview,
    ) -> ReassembledBleOcfPdu | None:
        data = _coerce_bytes(
            frame,
            name="BLE OCF frame",
            max_length=self._mtu,
            limit_name="MTU",
        )

        header = decode_header(data)
        if header.start:
            if self.in_progress:
                raise BleOcfInterleavedFrameError(
                    "received a new start frame while another PDU is incomplete"
                )
            return self._start(data, header)
        return self._continue(data, header)

    def _start(
        self,
        data: bytes,
        header: BleOcfHeader,
    ) -> ReassembledBleOcfPdu | None:
        if len(data) < BLE_FIRST_FRAME_OVERHEAD:
            raise BleOcfCodecError(
                "BLE OCF start frame is missing its four-byte length header"
            )

        expected_length = struct.unpack(
            ">I",
            data[BLE_HEADER_SIZE:BLE_FIRST_FRAME_OVERHEAD],
        )[0]
        if expected_length == 0:
            raise BleOcfCodecError("BLE OCF PDU length must not be zero")
        if expected_length > self._max_pdu_size:
            raise BleOcfCodecError(
                f"declared OCF PDU length {expected_length} exceeds maximum "
                f"{self._max_pdu_size}"
            )

        payload = data[BLE_FIRST_FRAME_OVERHEAD:]
        expected_payload_length = min(
            expected_length,
            self._mtu - BLE_FIRST_FRAME_OVERHEAD,
        )
        if len(payload) != expected_payload_length:
            raise BleOcfCodecError(
                "BLE OCF start frame payload length does not match "
                "IoTivity fragmentation"
            )

        if len(payload) == expected_length:
            return ReassembledBleOcfPdu(
                pdu=payload,
                source_port=header.source_port,
                destination_port=header.destination_port,
                secure=header.secure,
            )

        self._header = header
        self._expected_length = expected_length
        self._buffer.extend(payload)
        return None

    def _continue(
        self,
        data: bytes,
        header: BleOcfHeader,
    ) -> ReassembledBleOcfPdu | None:
        active_header = self._header
        if active_header is None:
            raise BleOcfCodecError(
                "received a continuation frame without a start frame"
            )
        if (
            header.start
            or header.source_port != active_header.source_port
            or header.secure != active_header.secure
            or header.destination_port != active_header.destination_port
        ):
            raise BleOcfInterleavedFrameError(
                "continuation frame metadata does not match the active PDU"
            )

        payload = data[BLE_HEADER_SIZE:]
        remaining = self._expected_length - len(self._buffer)
        expected_payload_length = min(
            remaining,
            self._mtu - BLE_HEADER_SIZE,
        )
        if len(payload) != expected_payload_length:
            raise BleOcfCodecError(
                "BLE OCF continuation payload length does not match "
                "IoTivity fragmentation"
            )

        self._buffer.extend(payload)
        if len(self._buffer) < self._expected_length:
            return None

        message = ReassembledBleOcfPdu(
            pdu=bytes(self._buffer),
            source_port=active_header.source_port,
            destination_port=active_header.destination_port,
            secure=active_header.secure,
        )
        self.reset()
        return message


class AdaptiveBleOcfReassembler:
    """Infer IoTivity's transport frame size from each first frame.

    A GATT client receives complete characteristic values, not necessarily the
    ATT MTU negotiated by the remote IoTivity stack. IoTivity fills an
    incomplete first frame to its usable frame size, so the observed length is
    the value needed by :class:`BleOcfReassembler`. A complete single-frame
    PDU is valid with that same observed length. This wrapper preserves strict
    fragmentation checks without trusting a platform-specific MTU property.
    """

    def __init__(
        self,
        *,
        max_pdu_size: int = DEFAULT_MAX_PDU_SIZE,
    ) -> None:
        self._max_pdu_size = _validate_max_pdu_size(max_pdu_size)
        self._reassembler: BleOcfReassembler | None = None

    @property
    def in_progress(self) -> bool:
        """Return whether a partial PDU is currently buffered."""

        return bool(self._reassembler and self._reassembler.in_progress)

    @property
    def buffered_bytes(self) -> int:
        """Return the number of PDU bytes retained for reassembly."""

        return self._reassembler.buffered_bytes if self._reassembler else 0

    def reset(self) -> None:
        """Discard any partial PDU and its inferred frame size."""

        self._reassembler = None

    def feed(
        self,
        frame: bytes | bytearray | memoryview,
    ) -> ReassembledBleOcfPdu | None:
        """Consume one frame, inferring a new frame size at PDU boundaries."""

        try:
            data = _coerce_bytes(
                frame,
                name="BLE OCF frame",
                max_length=BLE_MAX_MTU,
                limit_name="wire maximum",
            )
            if self._reassembler is None:
                header = decode_header(data)
                if not header.start:
                    raise BleOcfCodecError(
                        "received a continuation frame without a start frame"
                    )
                self._reassembler = BleOcfReassembler(
                    mtu=len(data),
                    max_pdu_size=self._max_pdu_size,
                )
            completed = self._reassembler.feed(data)
        except (BleOcfCodecError, TypeError):
            self.reset()
            raise
        if completed is not None:
            self.reset()
        return completed
