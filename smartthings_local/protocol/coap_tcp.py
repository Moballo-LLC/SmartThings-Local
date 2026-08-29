"""Pure CoAP-over-TCP wire codec used by Samsung's IoTivity stack.

IoTivity routes its TCP, GATT-BTLE, and RFCOMM adapters through the same
reliable-transport CoAP serializer.  The variable-length header follows the
format later standardized by RFC 8323: ``Len`` counts only encoded options and
the optional payload marker/payload.  Code and Token follow the length field
but are not included in ``Len``.

The framing behavior is also present in Samsung's public TizenRT copy of
IoTivity 1.2, notably ``CAGeneratePDUImpl``, ``coap_add_length``, and
``coap_get_total_message_length``.  This module implements only that pure wire
format.  It performs no socket, TLS, Bluetooth, account, or appliance I/O.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from .coap import (
    ACCEPT,
    CONTENT_FORMAT,
    METHOD_DELETE,
    METHOD_GET,
    METHOD_POST,
    URI_PATH,
    URI_QUERY,
)

__all__ = [
    "CoapTcpCodecError",
    "CoapTcpMessage",
    "CoapTcpStreamDecoder",
    "build_coap_tcp_csm",
    "build_coap_tcp_delete",
    "build_coap_tcp_get",
    "build_coap_tcp_message",
    "build_coap_tcp_post",
    "encode_uint_option",
    "parse_coap_tcp_message",
]

COAP_TCP_SHORT_LENGTH_LIMIT: Final = 13
COAP_TCP_8BIT_LENGTH_BASE: Final = 13
COAP_TCP_16BIT_LENGTH_BASE: Final = 269
COAP_TCP_32BIT_LENGTH_BASE: Final = 65_805
COAP_TCP_MAX_DECLARED_LENGTH: Final = 0xFFFFFFFF + COAP_TCP_32BIT_LENGTH_BASE
COAP_MAX_TOKEN_LENGTH: Final = 8
COAP_TCP_MAX_MESSAGE_SIZE: Final = (
    COAP_TCP_MAX_DECLARED_LENGTH + 5 + 1 + COAP_MAX_TOKEN_LENGTH
)
COAP_MAX_OPTION_NUMBER: Final = 0xFFFF
COAP_MAX_OPTION_VALUE_LENGTH: Final = 65_804
DEFAULT_MAX_MESSAGE_SIZE: Final = 4 * 1024 * 1024
PAYLOAD_MARKER: Final = 0xFF
CSM_CODE: Final = 0xE1
CSM_MAX_MESSAGE_SIZE_OPTION: Final = 2
CSM_BLOCK_WISE_TRANSFER_OPTION: Final = 4


class CoapTcpCodecError(ValueError):
    """An invalid or unsupported CoAP-over-TCP message."""


@dataclass(frozen=True, slots=True, repr=False)
class CoapTcpMessage:
    """One decoded CoAP-over-TCP message."""

    code: int
    token: bytes
    options: tuple[tuple[int, bytes], ...]
    payload: bytes

    def __repr__(self) -> str:
        """Return metadata without exposing token, option, or payload bytes."""
        return (
            "CoapTcpMessage("
            f"code={self.code!r}, option_count={len(self.options)}, "
            f"payload_length={len(self.payload)})"
        )


def _coerce_bytes(
    value: object,
    *,
    name: str,
    max_length: int | None = None,
) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    if max_length is not None and len(value) > max_length:
        raise CoapTcpCodecError(
            f"{name} length {len(value)} exceeds maximum {max_length}"
        )
    return bytes(value)


def _validate_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise CoapTcpCodecError(
            f"{name} must be in the range {minimum}..{maximum}"
        )
    return value


def _validate_max_message_size(value: object) -> int:
    return _validate_int(
        value,
        name="max_message_size",
        minimum=2,
        maximum=COAP_TCP_MAX_MESSAGE_SIZE,
    )


def encode_uint_option(value: int) -> bytes:
    """Encode a non-negative CoAP uint option in its shortest form."""
    value = _validate_int(
        value,
        name="option integer",
        minimum=0,
        maximum=0xFFFFFFFF,
    )
    if value == 0:
        return b""
    return value.to_bytes((value.bit_length() + 7) // 8, "big")


def _encode_extended(value: int, *, name: str) -> tuple[int, bytes]:
    if value < 13:
        return value, b""
    if value < 269:
        return 13, bytes((value - 13,))
    if value <= COAP_MAX_OPTION_VALUE_LENGTH:
        return 14, (value - 269).to_bytes(2, "big")
    raise CoapTcpCodecError(f"{name} is too large for a CoAP option header")


def _normalize_options(
    options: Iterable[tuple[int, bytes | bytearray | memoryview]],
) -> tuple[tuple[int, bytes], ...]:
    normalized: list[tuple[int, bytes]] = []
    for index, item in enumerate(options):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"option {index} must be a (number, value) pair")
        number = _validate_int(
            item[0],
            name=f"option {index} number",
            minimum=0,
            maximum=COAP_MAX_OPTION_NUMBER,
        )
        value = _coerce_bytes(
            item[1],
            name=f"option {index} value",
            max_length=COAP_MAX_OPTION_VALUE_LENGTH,
        )
        normalized.append((number, value))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _encode_options(options: tuple[tuple[int, bytes], ...]) -> bytes:
    encoded = bytearray()
    previous = 0
    for number, value in options:
        delta_nibble, delta_extra = _encode_extended(
            number - previous,
            name="option delta",
        )
        length_nibble, length_extra = _encode_extended(
            len(value),
            name="option length",
        )
        encoded.append((delta_nibble << 4) | length_nibble)
        encoded.extend(delta_extra)
        encoded.extend(length_extra)
        encoded.extend(value)
        previous = number
    return bytes(encoded)


def _encode_length(option_payload_length: int, token_length: int) -> bytes:
    if option_payload_length < COAP_TCP_SHORT_LENGTH_LIMIT:
        return bytes(((option_payload_length << 4) | token_length,))
    if option_payload_length < COAP_TCP_16BIT_LENGTH_BASE:
        return bytes(
            (
                (13 << 4) | token_length,
                option_payload_length - COAP_TCP_8BIT_LENGTH_BASE,
            )
        )
    if option_payload_length < COAP_TCP_32BIT_LENGTH_BASE:
        return bytes(((14 << 4) | token_length,)) + (
            option_payload_length - COAP_TCP_16BIT_LENGTH_BASE
        ).to_bytes(2, "big")
    if option_payload_length <= COAP_TCP_MAX_DECLARED_LENGTH:
        return bytes(((15 << 4) | token_length,)) + (
            option_payload_length - COAP_TCP_32BIT_LENGTH_BASE
        ).to_bytes(4, "big")
    raise CoapTcpCodecError(
        "declared CoAP-over-TCP length exceeds its wire field"
    )


def build_coap_tcp_message(
    *,
    code: int,
    token: bytes | bytearray | memoryview = b"",
    options: Iterable[tuple[int, bytes | bytearray | memoryview]] = (),
    payload: bytes | bytearray | memoryview = b"",
    max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
) -> bytes:
    """Build one bounded CoAP-over-TCP wire message."""
    maximum = _validate_max_message_size(max_message_size)
    code = _validate_int(code, name="code", minimum=0, maximum=0xFF)
    token_bytes = _coerce_bytes(
        token,
        name="token",
        max_length=COAP_MAX_TOKEN_LENGTH,
    )
    payload_bytes = _coerce_bytes(
        payload,
        name="payload",
        max_length=maximum,
    )
    encoded_options = _encode_options(_normalize_options(options))
    option_payload_length = len(encoded_options)
    if payload_bytes:
        option_payload_length += 1 + len(payload_bytes)

    prefix = _encode_length(option_payload_length, len(token_bytes))
    wire_size = len(prefix) + 1 + len(token_bytes) + option_payload_length
    if wire_size > maximum:
        raise CoapTcpCodecError(
            f"CoAP-over-TCP message length {wire_size} exceeds maximum "
            f"{maximum}"
        )

    wire = prefix + bytes((code,)) + token_bytes + encoded_options
    if payload_bytes:
        wire += bytes((PAYLOAD_MARKER,)) + payload_bytes
    return wire


def _path_options(
    path: str,
    query: Iterable[str],
) -> list[tuple[int, bytes]]:
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path.startswith("/") or "?" in path or "#" in path:
        raise CoapTcpCodecError(
            "path must be an absolute href without query or fragment"
        )
    if isinstance(query, (str, bytes, bytearray, memoryview)):
        raise TypeError("query must be an iterable of strings")

    options = [
        (URI_PATH, segment.encode("utf-8"))
        for segment in path.split("/")
        if segment
    ]
    for index, value in enumerate(query):
        if not isinstance(value, str):
            raise TypeError(f"query item {index} must be a string")
        if not value:
            raise CoapTcpCodecError(f"query item {index} must not be empty")
        options.append((URI_QUERY, value.encode("utf-8")))
    return options


def build_coap_tcp_get(
    path: str,
    *,
    token: bytes | bytearray | memoryview = b"",
    query: Iterable[str] = (),
    accept: int | None = None,
    extra_options: Iterable[
        tuple[int, bytes | bytearray | memoryview]
    ] = (),
    max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
) -> bytes:
    """Build a GET with repeated Uri-Path and Uri-Query options."""
    options = _path_options(path, query)
    if accept is not None:
        options.append((ACCEPT, encode_uint_option(accept)))
    options.extend(extra_options)
    return build_coap_tcp_message(
        code=METHOD_GET,
        token=token,
        options=options,
        max_message_size=max_message_size,
    )


def build_coap_tcp_post(
    path: str,
    payload: bytes | bytearray | memoryview,
    *,
    token: bytes | bytearray | memoryview = b"",
    query: Iterable[str] = (),
    content_format: int = 60,
    accept: int | None = 60,
    extra_options: Iterable[
        tuple[int, bytes | bytearray | memoryview]
    ] = (),
    max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
) -> bytes:
    """Build a POST with one bounded payload for an absolute OCF href."""
    options = _path_options(path, query)
    options.append((CONTENT_FORMAT, encode_uint_option(content_format)))
    if accept is not None:
        options.append((ACCEPT, encode_uint_option(accept)))
    options.extend(extra_options)
    return build_coap_tcp_message(
        code=METHOD_POST,
        token=token,
        options=options,
        payload=payload,
        max_message_size=max_message_size,
    )


def build_coap_tcp_csm(
    *,
    receive_max_message_size: int | None = None,
    block_wise_transfer: bool = False,
    max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
) -> bytes:
    """Build an RFC 8323 Capabilities and Settings Message.

    ``receive_max_message_size`` advertises the largest complete wire message
    this endpoint accepts. Omitting it retains RFC 8323's 1152-byte base value.
    ``block_wise_transfer`` advertises RFC 7959 support; paired with a value
    above 1152 it also advertises BERT support.
    """
    if not isinstance(block_wise_transfer, bool):
        raise TypeError("block_wise_transfer must be a boolean")
    options: list[tuple[int, bytes]] = []
    if receive_max_message_size is not None:
        receive_max_message_size = _validate_int(
            receive_max_message_size,
            name="receive_max_message_size",
            minimum=2,
            maximum=0xFFFFFFFF,
        )
        options.append(
            (
                CSM_MAX_MESSAGE_SIZE_OPTION,
                encode_uint_option(receive_max_message_size),
            )
        )
    if block_wise_transfer:
        options.append((CSM_BLOCK_WISE_TRANSFER_OPTION, b""))
    return build_coap_tcp_message(
        code=CSM_CODE,
        options=options,
        max_message_size=max_message_size,
    )


def build_coap_tcp_delete(
    path: str,
    *,
    token: bytes | bytearray | memoryview = b"",
    query: Iterable[str] = (),
    accept: int | None = None,
    extra_options: Iterable[
        tuple[int, bytes | bytearray | memoryview]
    ] = (),
    max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
) -> bytes:
    """Build a DELETE for one absolute OCF href and bounded query."""
    options = _path_options(path, query)
    if accept is not None:
        options.append((ACCEPT, encode_uint_option(accept)))
    options.extend(extra_options)
    return build_coap_tcp_message(
        code=METHOD_DELETE,
        token=token,
        options=options,
        max_message_size=max_message_size,
    )


def _decode_length(data: bytes | bytearray | memoryview) -> tuple[int, int, int]:
    if not data:
        raise CoapTcpCodecError("CoAP-over-TCP message is empty")
    first = data[0]
    length_nibble = first >> 4
    token_length = first & 0x0F
    if token_length > COAP_MAX_TOKEN_LENGTH:
        raise CoapTcpCodecError("CoAP token length exceeds 8 bytes")
    if length_nibble < 13:
        return length_nibble, token_length, 1

    extension_size = {13: 1, 14: 2, 15: 4}[length_nibble]
    prefix_size = 1 + extension_size
    if len(data) < prefix_size:
        raise CoapTcpCodecError("truncated CoAP-over-TCP length field")
    extension = int.from_bytes(data[1:prefix_size], "big")
    base = {
        13: COAP_TCP_8BIT_LENGTH_BASE,
        14: COAP_TCP_16BIT_LENGTH_BASE,
        15: COAP_TCP_32BIT_LENGTH_BASE,
    }[length_nibble]
    return base + extension, token_length, prefix_size


def _frame_size_from_prefix(
    data: bytes | bytearray | memoryview,
    *,
    maximum: int,
) -> int | None:
    """Return one frame size, or ``None`` until its length field is complete."""
    if not data:
        return None
    first = data[0]
    token_length = first & 0x0F
    if token_length > COAP_MAX_TOKEN_LENGTH:
        raise CoapTcpCodecError("CoAP token length exceeds 8 bytes")
    length_nibble = first >> 4
    extension_size = 0 if length_nibble < 13 else {13: 1, 14: 2, 15: 4}[
        length_nibble
    ]
    prefix_size = 1 + extension_size
    if len(data) < prefix_size:
        return None
    option_payload_length, _, _ = _decode_length(data[:prefix_size])
    frame_size = prefix_size + 1 + token_length + option_payload_length
    if frame_size > maximum:
        raise CoapTcpCodecError(
            f"declared CoAP-over-TCP message length {frame_size} exceeds "
            f"maximum {maximum}"
        )
    return frame_size


def _decode_extended(
    data: bytes,
    cursor: int,
    nibble: int,
) -> tuple[int, int]:
    if nibble < 13:
        return nibble, cursor
    if nibble == 15:
        raise CoapTcpCodecError("reserved CoAP option nibble 15")
    extension_size = 1 if nibble == 13 else 2
    end = cursor + extension_size
    if end > len(data):
        raise CoapTcpCodecError("truncated CoAP option extension")
    base = 13 if nibble == 13 else 269
    return base + int.from_bytes(data[cursor:end], "big"), end


def _decode_options(
    data: bytes,
) -> tuple[tuple[tuple[int, bytes], ...], bytes]:
    options: list[tuple[int, bytes]] = []
    previous = 0
    cursor = 0
    while cursor < len(data):
        first = data[cursor]
        cursor += 1
        if first == PAYLOAD_MARKER:
            if cursor == len(data):
                raise CoapTcpCodecError("CoAP payload marker has no payload")
            return tuple(options), data[cursor:]

        delta, cursor = _decode_extended(data, cursor, first >> 4)
        length, cursor = _decode_extended(data, cursor, first & 0x0F)
        number = previous + delta
        if number > COAP_MAX_OPTION_NUMBER:
            raise CoapTcpCodecError(
                "decoded CoAP option number exceeds 65535"
            )
        end = cursor + length
        if end > len(data):
            raise CoapTcpCodecError("truncated CoAP option value")
        options.append((number, data[cursor:end]))
        cursor = end
        previous = number
    return tuple(options), b""


def parse_coap_tcp_message(
    data: bytes | bytearray | memoryview,
    *,
    max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
) -> CoapTcpMessage:
    """Parse exactly one complete bounded CoAP-over-TCP message."""
    maximum = _validate_max_message_size(max_message_size)
    wire = _coerce_bytes(data, name="data", max_length=maximum)

    option_payload_length, token_length, prefix_size = _decode_length(wire)
    expected_size = prefix_size + 1 + token_length + option_payload_length
    if expected_size > maximum:
        raise CoapTcpCodecError(
            f"declared CoAP-over-TCP message length {expected_size} exceeds "
            f"maximum {maximum}"
        )
    if len(wire) < expected_size:
        raise CoapTcpCodecError("truncated CoAP-over-TCP message body")
    if len(wire) > expected_size:
        raise CoapTcpCodecError("trailing bytes after CoAP-over-TCP message")

    code_offset = prefix_size
    token_offset = code_offset + 1
    option_offset = token_offset + token_length
    options, payload = _decode_options(wire[option_offset:])
    return CoapTcpMessage(
        code=wire[code_offset],
        token=wire[token_offset:option_offset],
        options=options,
        payload=payload,
    )


class CoapTcpStreamDecoder:
    """Incrementally split and parse CoAP messages from a byte stream.

    Only one incomplete frame is buffered. Complete concatenated frames are
    emitted immediately, so a large read containing many small valid messages
    is not treated as one oversized message.
    """

    def __init__(
        self,
        *,
        max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
    ) -> None:
        self._max_message_size = _validate_max_message_size(max_message_size)
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        """Return the number of bytes retained for one incomplete frame."""
        return len(self._buffer)

    def reset(self) -> None:
        """Discard an incomplete frame."""
        self._buffer.clear()

    def feed(
        self,
        data: bytes | bytearray | memoryview,
    ) -> tuple[CoapTcpMessage, ...]:
        """Consume a stream chunk and return every complete message in it."""
        chunk = _coerce_bytes(data, name="data")
        messages: list[CoapTcpMessage] = []
        offset = 0
        try:
            while offset < len(chunk):
                if self._buffer:
                    frame_size = _frame_size_from_prefix(
                        self._buffer,
                        maximum=self._max_message_size,
                    )
                    if frame_size is None:
                        self._buffer.append(chunk[offset])
                        offset += 1
                        continue
                    take = min(
                        frame_size - len(self._buffer),
                        len(chunk) - offset,
                    )
                    self._buffer.extend(chunk[offset:offset + take])
                    offset += take
                    if len(self._buffer) < frame_size:
                        break
                    frame = bytes(self._buffer)
                    self._buffer.clear()
                else:
                    remaining = memoryview(chunk)[offset:]
                    frame_size = _frame_size_from_prefix(
                        remaining,
                        maximum=self._max_message_size,
                    )
                    if frame_size is None or len(remaining) < frame_size:
                        self._buffer.extend(remaining)
                        break
                    frame = bytes(remaining[:frame_size])
                    offset += frame_size

                messages.append(
                    parse_coap_tcp_message(
                        frame,
                        max_message_size=self._max_message_size,
                    )
                )
        except (CoapTcpCodecError, TypeError):
            self.reset()
            raise
        return tuple(messages)

    def finish(self) -> None:
        """Accept end-of-stream only when no partial message remains."""
        if self._buffer:
            self.reset()
            raise CoapTcpCodecError(
                "truncated CoAP-over-TCP message at end of stream"
            )
