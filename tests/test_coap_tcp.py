"""CoAP-over-TCP framing and stream-decoding contracts."""

from __future__ import annotations

import pytest

from smartthings_local.protocol.coap_tcp import (
    COAP_TCP_MAX_MESSAGE_SIZE,
    CoapTcpCodecError,
    CoapTcpMessage,
    CoapTcpStreamDecoder,
    build_coap_tcp_csm,
    build_coap_tcp_delete,
    build_coap_tcp_get,
    build_coap_tcp_message,
    build_coap_tcp_post,
    encode_uint_option,
    parse_coap_tcp_message,
)


def test_delete_round_trip_preserves_repeated_query_and_extension_options():
    wire = build_coap_tcp_delete(
        "/oic/sec/cred",
        token=b"\x12\x34",
        query=("credid=7", "credid=19"),
        extra_options=((65524, b"\xc0"),),
    )

    message = parse_coap_tcp_message(wire)

    assert message == CoapTcpMessage(
        code=0x04,
        token=b"\x12\x34",
        options=(
            (11, b"oic"),
            (11, b"sec"),
            (11, b"cred"),
            (15, b"credid=7"),
            (15, b"credid=19"),
            (65524, b"\xc0"),
        ),
        payload=b"",
    )


def test_post_round_trip_uses_query_cbor_content_format_and_accept():
    wire = build_coap_tcp_post(
        "/oic/sec/doxm",
        b"\xa1eowned\xf5",
        token=b"\x11\x22",
        query=("if=oic.if.rw",),
    )

    message = parse_coap_tcp_message(wire)

    assert message.code == 0x02
    assert message.token == b"\x11\x22"
    assert message.options == (
        (11, b"oic"),
        (11, b"sec"),
        (11, b"doxm"),
        (12, b"<"),
        (15, b"if=oic.if.rw"),
        (17, b"<"),
    )
    assert message.payload == b"\xa1eowned\xf5"


def test_rfc8323_length_excludes_code_and_token():
    wire = build_coap_tcp_get("/oic/res", token=b"\xaa\xbb")

    # Len=8 is exactly the two Uri-Path option encodings. Code and Token
    # follow the length field but are not included in the declared length.
    assert wire == bytes.fromhex("82 01 aa bb b3 6f6963 03 726573")
    assert parse_coap_tcp_message(wire) == CoapTcpMessage(
        code=0x01,
        token=b"\xaa\xbb",
        options=((11, b"oic"), (11, b"res")),
        payload=b"",
    )

    assert build_coap_tcp_message(code=0x01) == b"\x00\x01"
    with pytest.raises(CoapTcpCodecError, match="truncated"):
        parse_coap_tcp_message(b"\x10\x01")


def test_source_derived_ocf_response_vector_round_trips():
    wire = bytes.fromhex("81 45 aa c2 2710 ff a1617801")
    message = CoapTcpMessage(
        code=0x45,
        token=b"\xaa",
        options=((12, bytes.fromhex("2710")),),
        payload=bytes.fromhex("a1617801"),
    )

    assert parse_coap_tcp_message(wire) == message
    assert build_coap_tcp_message(
        code=message.code,
        token=message.token,
        options=message.options,
        payload=message.payload,
    ) == wire


def test_iotivity_csm_vector_uses_reliable_transport_header():
    assert build_coap_tcp_csm(
        receive_max_message_size=1152,
    ) == bytes.fromhex("30 e1 22 0480")

    message = parse_coap_tcp_message(
        build_coap_tcp_csm(
            receive_max_message_size=4096,
            block_wise_transfer=True,
        )
    )
    assert message == CoapTcpMessage(
        code=0xE1,
        token=b"",
        options=((2, b"\x10\x00"), (4, b"")),
        payload=b"",
    )


def test_empty_csm_uses_rfc_base_capabilities():
    assert build_coap_tcp_csm() == b"\x00\xe1"


@pytest.mark.parametrize(
    ("declared_length", "prefix"),
    (
        (0, bytes.fromhex("00")),
        (12, bytes.fromhex("c0")),
        (13, bytes.fromhex("d0 00")),
        (268, bytes.fromhex("d0 ff")),
        (269, bytes.fromhex("e0 0000")),
        (65_804, bytes.fromhex("e0 ffff")),
        (65_805, bytes.fromhex("f0 00000000")),
    ),
)
def test_iotivity_length_field_boundaries(declared_length, prefix):
    payload = b"" if declared_length == 0 else b"x" * (declared_length - 1)

    wire = build_coap_tcp_message(
        code=0x45,
        payload=payload,
        max_message_size=70_000,
    )

    assert wire[:len(prefix)] == prefix
    assert parse_coap_tcp_message(
        wire,
        max_message_size=70_000,
    ).payload == payload


def test_get_builder_encodes_query_accept_utf8_and_extra_options():
    wire = build_coap_tcp_get(
        "/a/temperature",
        token=memoryview(b"12345678"),
        query=("if=oic.if.a", "lang=caf\N{LATIN SMALL LETTER E WITH ACUTE}"),
        accept=60,
        extra_options=((6, b""),),
    )

    message = parse_coap_tcp_message(wire)

    assert message.code == 0x01
    assert message.token == b"12345678"
    assert message.options == (
        (6, b""),
        (11, b"a"),
        (11, b"temperature"),
        (15, b"if=oic.if.a"),
        (15, "lang=caf\N{LATIN SMALL LETTER E WITH ACUTE}".encode()),
        (17, b"<"),
    )


def test_extended_options_and_duplicate_numbers_round_trip():
    options = (
        (13, b"a" * 13),
        (13, b""),
        (300, b"z" * 269),
        (65_535, b"tail"),
    )

    wire = build_coap_tcp_message(code=0x45, options=options)

    assert parse_coap_tcp_message(wire).options == options


@pytest.mark.parametrize(
    ("wire", "error"),
    (
        (b"", "empty"),
        (b"\xd0", "length field"),
        (b"\x00", "body"),
        (b"\x10\x01", "body"),
        (b"\x00\x01\x00", "trailing"),
        (b"\x10\x45\xff", "marker"),
        (b"\x10\x45\xf0", "nibble"),
        (b"\x10\x45\xd0", "extension"),
        (b"\x10\x45\x01", "value"),
        (b"\x09\x01" + b"x" * 9, "token length"),
    ),
)
def test_parser_rejects_truncation_trailing_and_invalid_options(wire, error):
    with pytest.raises(CoapTcpCodecError, match=error):
        parse_coap_tcp_message(wire)


def test_declared_and_actual_message_bounds_fail_closed():
    with pytest.raises(CoapTcpCodecError, match="declared.*exceeds"):
        parse_coap_tcp_message(
            bytes.fromhex("f0 00000000 45"),
            max_message_size=100,
        )
    with pytest.raises(CoapTcpCodecError, match="exceeds maximum"):
        build_coap_tcp_message(
            code=1,
            payload=b"x" * 20,
            max_message_size=10,
        )
    assert build_coap_tcp_message(
        code=1,
        max_message_size=COAP_TCP_MAX_MESSAGE_SIZE,
    ) == b"\x00\x01"
    with pytest.raises(CoapTcpCodecError, match="max_message_size"):
        build_coap_tcp_message(
            code=1,
            max_message_size=COAP_TCP_MAX_MESSAGE_SIZE + 1,
        )


def test_builder_rejects_unsafe_or_ambiguous_inputs():
    with pytest.raises(CoapTcpCodecError):
        build_coap_tcp_message(code=256)
    with pytest.raises(CoapTcpCodecError):
        build_coap_tcp_message(code=1, token=b"123456789")
    with pytest.raises(TypeError):
        build_coap_tcp_message(code=True)
    with pytest.raises(TypeError):
        build_coap_tcp_message(code=1, token="not-bytes")
    with pytest.raises(TypeError):
        build_coap_tcp_message(code=1, options=((11, "not-bytes"),))
    with pytest.raises(CoapTcpCodecError):
        build_coap_tcp_message(code=1, options=((65_536, b"x"),))
    with pytest.raises(CoapTcpCodecError, match="option 0 value"):
        build_coap_tcp_message(code=1, options=((11, memoryview(b"x" * 65_805)),))

    for path in ("relative", "/has?query", "/has#fragment"):
        with pytest.raises(CoapTcpCodecError):
            build_coap_tcp_get(path)
    with pytest.raises(TypeError):
        build_coap_tcp_get("/path", query="if=oic.if.a")
    with pytest.raises(TypeError):
        build_coap_tcp_get("/path", query=(1,))
    with pytest.raises(CoapTcpCodecError):
        build_coap_tcp_get("/path", query=("",))
    with pytest.raises(TypeError):
        build_coap_tcp_csm(block_wise_transfer=1)
    with pytest.raises(CoapTcpCodecError):
        build_coap_tcp_csm(receive_max_message_size=1)
    with pytest.raises(CoapTcpCodecError):
        build_coap_tcp_csm(receive_max_message_size=0x1_0000_0000)


def test_uint_option_uses_minimal_network_order():
    assert encode_uint_option(0) == b""
    assert encode_uint_option(60) == b"\x3c"
    assert encode_uint_option(10_000) == bytes.fromhex("2710")
    with pytest.raises(CoapTcpCodecError):
        encode_uint_option(-1)


def test_message_repr_does_not_expose_wire_values():
    message = CoapTcpMessage(
        code=0x45,
        token=b"token-not-for-repr",
        options=((11, b"option-not-for-repr"),),
        payload=b"payload-not-for-repr",
    )

    rendered = repr(message)

    assert rendered == "CoapTcpMessage(code=69, option_count=1, payload_length=20)"
    assert "token-not-for-repr" not in rendered
    assert "option-not-for-repr" not in rendered
    assert "payload-not-for-repr" not in rendered


def test_stream_decoder_accepts_every_two_chunk_split_and_concatenation():
    wires = (
        build_coap_tcp_get("/oic/res", token=b"a"),
        build_coap_tcp_post("/mode/vs/0", b"payload", token=b"bc"),
        build_coap_tcp_delete("/oic/sec/cred", token=b"def"),
    )
    stream = b"".join(wires)
    expected = tuple(parse_coap_tcp_message(wire) for wire in wires)

    for split in range(len(stream) + 1):
        decoder = CoapTcpStreamDecoder()
        messages = decoder.feed(stream[:split]) + decoder.feed(stream[split:])
        decoder.finish()
        assert messages == expected
        assert decoder.buffered_bytes == 0


def test_stream_decoder_accepts_bytewise_extended_length_input():
    wire = build_coap_tcp_message(code=0x45, payload=b"x" * 300)
    decoder = CoapTcpStreamDecoder()
    messages = ()

    for value in wire:
        messages += decoder.feed(bytes((value,)))

    decoder.finish()
    assert messages == (parse_coap_tcp_message(wire),)


def test_stream_decoder_does_not_apply_one_frame_bound_to_whole_chunk():
    wire = build_coap_tcp_message(code=0x45)
    decoder = CoapTcpStreamDecoder(max_message_size=len(wire))

    messages = decoder.feed(wire * 100)

    assert len(messages) == 100
    assert all(message.code == 0x45 for message in messages)
    assert decoder.buffered_bytes == 0


def test_stream_decoder_rejects_oversize_from_prefix_and_resets():
    decoder = CoapTcpStreamDecoder(max_message_size=100)

    with pytest.raises(CoapTcpCodecError, match="declared.*exceeds"):
        decoder.feed(bytes.fromhex("f0 00000000"))

    assert decoder.buffered_bytes == 0
    assert decoder.feed(build_coap_tcp_message(code=0x45)) == (
        CoapTcpMessage(code=0x45, token=b"", options=(), payload=b""),
    )


def test_stream_decoder_rejects_malformed_frame_and_can_recover():
    decoder = CoapTcpStreamDecoder()

    with pytest.raises(CoapTcpCodecError, match="marker"):
        decoder.feed(b"\x10\x45\xff")

    assert decoder.buffered_bytes == 0
    assert len(decoder.feed(build_coap_tcp_message(code=0x45))) == 1


def test_stream_decoder_finish_rejects_and_discards_partial_frame():
    decoder = CoapTcpStreamDecoder()
    assert decoder.feed(b"\xe0\x00") == ()
    assert decoder.buffered_bytes == 2

    with pytest.raises(CoapTcpCodecError, match="end of stream"):
        decoder.finish()

    assert decoder.buffered_bytes == 0
