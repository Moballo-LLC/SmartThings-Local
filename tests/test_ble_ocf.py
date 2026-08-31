"""IoTivity BLE OCF fragmentation and reassembly contracts."""

from __future__ import annotations

from array import array

import pytest

from smartthings_local.protocol.ble_ocf import (
    BLE_MAX_MTU,
    BLE_WIRE_MAX_PDU_SIZE,
    AdaptiveBleOcfReassembler,
    BleOcfCodecError,
    BleOcfHeader,
    BleOcfInterleavedFrameError,
    BleOcfReassembler,
    ReassembledBleOcfPdu,
    decode_header,
    encode_header,
    fragment_pdu,
)
from smartthings_local.protocol.coap_tcp import (
    CoapTcpMessage,
    build_coap_tcp_get,
    parse_coap_tcp_message,
)


@pytest.mark.parametrize(
    ("fields", "encoded"),
    (
        (
            {
                "start": True,
                "source_port": 1,
                "secure": False,
                "destination_port": 0,
            },
            b"\x81\x00",
        ),
        (
            {
                "start": False,
                "source_port": 127,
                "secure": True,
                "destination_port": 42,
            },
            b"\x7f\xaa",
        ),
        (
            {
                "start": True,
                "source_port": 37,
                "secure": True,
                "destination_port": 127,
            },
            b"\xa5\xff",
        ),
    ),
)
def test_header_known_vectors_preserve_flags_and_ports(fields, encoded):
    assert encode_header(**fields) == encoded
    assert decode_header(encoded + b"ignored payload") == BleOcfHeader(**fields)


def test_fragmentation_source_vector_uses_big_endian_length_and_mtu():
    assert fragment_pdu(
        b"0123456789ABC",
        mtu=10,
        source_port=5,
        destination_port=9,
        secure=True,
    ) == (
        b"\x85\x89\x00\x00\x00\x0d0123",
        b"\x05\x89456789AB",
        b"\x05\x89C",
    )


@pytest.mark.parametrize("mtu", (7, 8, 10, 20, 23, 64, 512))
@pytest.mark.parametrize("secure", (False, True))
def test_round_trip_at_every_fragment_boundary(mtu, secure):
    first_capacity = mtu - 6
    continuation_capacity = mtu - 2
    sizes = {
        1,
        first_capacity,
        first_capacity + 1,
        first_capacity + continuation_capacity,
        first_capacity + continuation_capacity + 1,
        first_capacity + 3 * continuation_capacity,
        first_capacity + 3 * continuation_capacity + 1,
    }

    for size in sorted(sizes):
        pdu = bytes(index % 251 for index in range(size))
        frames = fragment_pdu(
            pdu,
            mtu=mtu,
            source_port=17,
            destination_port=0,
            secure=secure,
        )
        reassembler = BleOcfReassembler(mtu=mtu)
        completed = None

        for frame in frames:
            assert len(frame) <= mtu
            completed = reassembler.feed(frame)

        assert completed == ReassembledBleOcfPdu(
            pdu=pdu,
            source_port=17,
            destination_port=0,
            secure=secure,
        )
        assert not reassembler.in_progress
        assert reassembler.buffered_bytes == 0


def test_non_final_frames_fill_mtu_including_exact_continuation_multiple():
    mtu = 20
    first_capacity = mtu - 6
    continuation_capacity = mtu - 2
    pdu = b"x" * (first_capacity + 3 * continuation_capacity)

    frames = fragment_pdu(
        pdu,
        mtu=mtu,
        source_port=1,
        destination_port=1,
        secure=False,
    )

    assert len(frames) == 4
    assert {len(frame) for frame in frames} == {mtu}


def test_tcp_get_round_trips_through_ble_transport_frames():
    pdu = build_coap_tcp_get(
        "/oic/res",
        token=b"\x12\x34",
        query=("if=oic.if.baseline",),
    )
    reassembler = BleOcfReassembler(mtu=20)

    for frame in fragment_pdu(
        pdu,
        mtu=20,
        source_port=1,
        destination_port=0,
        secure=False,
    ):
        completed = reassembler.feed(frame)

    assert completed is not None
    assert completed.destination_port == 0
    assert parse_coap_tcp_message(completed.pdu) == CoapTcpMessage(
        code=0x01,
        token=b"\x12\x34",
        options=(
            (11, b"oic"),
            (11, b"res"),
            (15, b"if=oic.if.baseline"),
        ),
        payload=b"",
    )


def test_bytes_like_inputs_are_snapshotted():
    mutable_pdu = bytearray(b"payload that fragments")
    frames = fragment_pdu(
        memoryview(mutable_pdu),
        mtu=10,
        source_port=3,
        destination_port=7,
        secure=False,
    )
    mutable_pdu[:] = b"z" * len(mutable_pdu)

    reassembler = BleOcfReassembler(mtu=10)
    completed = None
    for frame in (bytearray(value) for value in frames):
        completed = reassembler.feed(frame)

    assert completed is not None
    assert completed.pdu == b"payload that fragments"


def test_adaptive_reassembler_infers_each_first_frame_size():
    adaptive = AdaptiveBleOcfReassembler(max_pdu_size=1024)
    expected = (
        (b"first", 20),
        (b"second payload that spans frames", 11),
        (b"third", 23),
    )
    completed = []

    for pdu, mtu in expected:
        for frame in fragment_pdu(
            pdu,
            mtu=mtu,
            source_port=1,
            destination_port=1,
            secure=False,
            max_pdu_size=1024,
        ):
            result = adaptive.feed(frame)
            if result is not None:
                completed.append(result)

    assert completed == [
        ReassembledBleOcfPdu(
            pdu=pdu,
            source_port=1,
            destination_port=1,
            secure=False,
        )
        for pdu, _mtu in expected
    ]
    assert not adaptive.in_progress
    assert adaptive.buffered_bytes == 0


def test_adaptive_reassembler_resets_after_malformed_stream():
    adaptive = AdaptiveBleOcfReassembler(max_pdu_size=64)
    frames = fragment_pdu(
        b"long enough to fragment",
        mtu=10,
        source_port=1,
        destination_port=1,
        secure=False,
    )
    assert adaptive.feed(frames[0]) is None
    assert adaptive.buffered_bytes == 4

    with pytest.raises(BleOcfInterleavedFrameError):
        adaptive.feed(b"\x02\x01bad-tail")

    assert not adaptive.in_progress
    recovered = adaptive.feed(
        fragment_pdu(
            b"ok",
            mtu=20,
            source_port=1,
            destination_port=1,
            secure=False,
        )[0]
    )
    assert recovered is not None
    assert recovered.pdu == b"ok"


def test_adaptive_reassembler_resets_after_oversized_frame():
    adaptive = AdaptiveBleOcfReassembler(max_pdu_size=64)
    first = fragment_pdu(
        b"long enough to fragment",
        mtu=10,
        source_port=1,
        destination_port=1,
        secure=False,
    )[0]
    assert adaptive.feed(first) is None

    with pytest.raises(BleOcfCodecError, match="wire maximum"):
        adaptive.feed(b"x" * (BLE_MAX_MTU + 1))

    assert not adaptive.in_progress
    assert adaptive.buffered_bytes == 0


def test_released_memoryview_resets_partial_reassembly():
    reassembler = BleOcfReassembler(mtu=10)
    first = fragment_pdu(
        b"long enough to fragment",
        mtu=10,
        source_port=1,
        destination_port=1,
        secure=False,
    )[0]
    assert reassembler.feed(first) is None
    released = memoryview(b"unused")
    released.release()

    with pytest.raises(TypeError, match="active bytes-like"):
        reassembler.feed(released)

    assert not reassembler.in_progress
    assert reassembler.buffered_bytes == 0


@pytest.mark.parametrize("source_port", (0, 128, -1))
def test_header_rejects_invalid_source_port(source_port):
    with pytest.raises(BleOcfCodecError, match="source_port"):
        encode_header(
            start=True,
            source_port=source_port,
            secure=False,
            destination_port=0,
        )


@pytest.mark.parametrize("destination_port", (-1, 128))
def test_header_rejects_invalid_destination_port(destination_port):
    with pytest.raises(BleOcfCodecError, match="destination_port"):
        encode_header(
            start=True,
            source_port=1,
            secure=False,
            destination_port=destination_port,
        )


@pytest.mark.parametrize(("name", "value"), (("start", 1), ("secure", 0)))
def test_header_rejects_non_boolean_flags(name, value):
    fields = {
        "start": True,
        "source_port": 1,
        "secure": False,
        "destination_port": 0,
    }
    fields[name] = value
    with pytest.raises(TypeError, match="bool"):
        encode_header(**fields)


@pytest.mark.parametrize("mtu", (0, 6, BLE_MAX_MTU + 1))
def test_fragmenter_rejects_invalid_mtu(mtu):
    with pytest.raises(BleOcfCodecError, match="mtu"):
        fragment_pdu(
            b"x",
            mtu=mtu,
            source_port=1,
            destination_port=0,
            secure=False,
        )


@pytest.mark.parametrize("mtu", (True, 20.0, "20"))
def test_fragmenter_rejects_non_integer_mtu(mtu):
    with pytest.raises(TypeError, match="mtu"):
        fragment_pdu(
            b"x",
            mtu=mtu,
            source_port=1,
            destination_port=0,
            secure=False,
        )


def test_pdu_and_maximum_validation_is_bounded():
    with pytest.raises(BleOcfCodecError, match="must not be empty"):
        fragment_pdu(
            b"",
            mtu=20,
            source_port=1,
            destination_port=0,
            secure=False,
        )
    with pytest.raises(BleOcfCodecError, match="length 5 exceeds maximum 4"):
        fragment_pdu(
            b"12345",
            mtu=20,
            source_port=1,
            destination_port=0,
            secure=False,
            max_pdu_size=4,
        )
    with pytest.raises(TypeError, match="bytes-like"):
        fragment_pdu(
            "not bytes",
            mtu=20,
            source_port=1,
            destination_port=0,
            secure=False,
        )
    for maximum in (0, BLE_WIRE_MAX_PDU_SIZE + 1):
        with pytest.raises(BleOcfCodecError, match="max_pdu_size"):
            BleOcfReassembler(mtu=20, max_pdu_size=maximum)
    with pytest.raises(TypeError, match="max_pdu_size"):
        AdaptiveBleOcfReassembler(max_pdu_size=True)


def test_memoryview_bounds_use_bytes_not_element_count():
    wide_view = memoryview(array("I", (1, 2)))
    assert len(wide_view) == 2
    assert wide_view.nbytes > 2
    maximum = wide_view.nbytes - 1

    with pytest.raises(
        BleOcfCodecError,
        match=rf"length {wide_view.nbytes} exceeds maximum {maximum}",
    ):
        fragment_pdu(
            wide_view,
            mtu=20,
            source_port=1,
            destination_port=0,
            secure=False,
            max_pdu_size=maximum,
        )


@pytest.mark.parametrize(
    "frame",
    (
        b"",
        b"\x81",
        b"\x80\x00",  # source port zero
        b"\x81\x00",  # missing length field
        b"\x81\x00\x00\x00\x00\x00",  # zero-length PDU
        b"\x81\x00\x00\x00\x00\x02x",  # short completed start
        b"\x81\x00\x00\x00\x00\x01xy",  # payload exceeds declaration
        b"\x01\x00x",  # continuation without a start
    ),
)
def test_reassembler_rejects_malformed_frames_and_resets(frame):
    reassembler = BleOcfReassembler(mtu=10)

    with pytest.raises(BleOcfCodecError):
        reassembler.feed(frame)

    assert not reassembler.in_progress
    assert reassembler.buffered_bytes == 0


def test_reassembler_rejects_oversized_frame_and_declaration():
    with pytest.raises(BleOcfCodecError, match="exceeds MTU"):
        BleOcfReassembler(mtu=10).feed(b"\x81\x00\x00\x00\x00\x05abcde")
    with pytest.raises(BleOcfCodecError, match="declared.*exceeds maximum"):
        BleOcfReassembler(mtu=10, max_pdu_size=12).feed(b"\x81\x00\x00\x00\x00\x0d0123")
    with pytest.raises(BleOcfCodecError, match="wire maximum"):
        AdaptiveBleOcfReassembler().feed(b"x" * (BLE_MAX_MTU + 1))


def test_reassembler_rejects_short_middle_and_empty_continuations():
    frames = fragment_pdu(
        b"a" * 30,
        mtu=10,
        source_port=5,
        destination_port=9,
        secure=True,
    )

    for bad_continuation in (frames[1][:-1], frames[1][:2]):
        reassembler = BleOcfReassembler(mtu=10)
        assert reassembler.feed(frames[0]) is None
        with pytest.raises(BleOcfCodecError, match="continuation payload"):
            reassembler.feed(bad_continuation)
        assert not reassembler.in_progress


def test_reassembler_rejects_interleaved_start_and_resets():
    first = fragment_pdu(
        b"first-pdu",
        mtu=10,
        source_port=5,
        destination_port=9,
        secure=True,
    )
    second = fragment_pdu(
        b"second-pdu",
        mtu=10,
        source_port=6,
        destination_port=9,
        secure=True,
    )
    reassembler = BleOcfReassembler(mtu=10)

    assert reassembler.feed(first[0]) is None
    with pytest.raises(BleOcfInterleavedFrameError, match="new start"):
        reassembler.feed(second[0])

    assert not reassembler.in_progress


@pytest.mark.parametrize(
    "changed_header",
    (
        encode_header(
            start=False,
            source_port=6,
            destination_port=9,
            secure=True,
        ),
        encode_header(
            start=False,
            source_port=5,
            destination_port=8,
            secure=True,
        ),
        encode_header(
            start=False,
            source_port=5,
            destination_port=9,
            secure=False,
        ),
    ),
)
def test_reassembler_rejects_changed_continuation_metadata(changed_header):
    frames = fragment_pdu(
        b"first-pdu",
        mtu=10,
        source_port=5,
        destination_port=9,
        secure=True,
    )
    reassembler = BleOcfReassembler(mtu=10)

    assert reassembler.feed(frames[0]) is None
    with pytest.raises(BleOcfInterleavedFrameError, match="metadata"):
        reassembler.feed(changed_header + frames[1][2:])

    assert not reassembler.in_progress


def test_duplicate_start_and_completed_tail_fail_closed():
    frames = fragment_pdu(
        b"a payload long enough for several frames",
        mtu=10,
        source_port=5,
        destination_port=9,
        secure=False,
    )
    reassembler = BleOcfReassembler(mtu=10)

    assert reassembler.feed(frames[0]) is None
    with pytest.raises(BleOcfInterleavedFrameError):
        reassembler.feed(frames[0])

    completed = None
    for frame in frames:
        completed = reassembler.feed(frame)
    assert completed is not None
    with pytest.raises(BleOcfCodecError, match="without a start"):
        reassembler.feed(frames[-1])


def test_missing_fragment_is_incomplete_or_rejected_then_can_be_reset():
    frames = fragment_pdu(
        b"a" * 31,
        mtu=10,
        source_port=5,
        destination_port=9,
        secure=False,
    )
    reassembler = BleOcfReassembler(mtu=10)

    for frame in frames[:-1]:
        assert reassembler.feed(frame) is None
    assert reassembler.in_progress
    assert reassembler.buffered_bytes < 31
    reassembler.reset()
    assert not reassembler.in_progress

    assert reassembler.feed(frames[0]) is None
    with pytest.raises(BleOcfCodecError, match="continuation payload"):
        reassembler.feed(frames[-1])
    assert not reassembler.in_progress


def test_failed_stream_accepts_a_fresh_pdu():
    frames = fragment_pdu(
        b"recovered",
        mtu=10,
        source_port=3,
        destination_port=7,
        secure=False,
    )
    reassembler = BleOcfReassembler(mtu=10)
    with pytest.raises(BleOcfCodecError):
        reassembler.feed(b"\x03\x07bad")

    completed = None
    for frame in frames:
        completed = reassembler.feed(frame)
    assert completed is not None
    assert completed.pdu == b"recovered"


def test_reassembled_pdu_repr_omits_wire_bytes():
    result = ReassembledBleOcfPdu(
        pdu=b"token-and-payload-not-for-repr",
        source_port=1,
        destination_port=0,
        secure=False,
    )

    rendered = repr(result)

    assert rendered == (
        "ReassembledBleOcfPdu(pdu_length=30, source_port=1, "
        "destination_port=0, secure=False)"
    )
    assert "token-and-payload" not in rendered
