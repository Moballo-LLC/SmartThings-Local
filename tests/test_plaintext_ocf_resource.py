"""Bounded plaintext OCF resource reads stay source-correlated and raw."""

import inspect
import socket
import threading
import time
import traceback
from dataclasses import FrozenInstanceError

import cbor2
import pytest

from smartthings_local.errors import EndpointError
from smartthings_local.protocol import ocf_discovery as discovery
from smartthings_local.protocol.coap import (
    ACCEPT,
    BLOCK2,
    CF_CBOR,
    CONTENT_FORMAT,
    ETAG,
    METHOD_GET,
    SIZE2,
    TYPE_ACK,
    TYPE_CON,
    TYPE_NON,
    URI_PATH,
    URI_QUERY,
    block_value,
    build_coap,
    parse_coap,
)


def _option_map(options):
    result = {}
    for number, value in options:
        result.setdefault(number, []).append(value)
    return result


def _uint_bytes(value):
    length = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(length, 'big')


def test_public_exports_and_signature_are_explicit():
    assert 'PlaintextOcfResourceResult' in discovery.__all__
    assert 'read_plaintext_ocf_resource' in discovery.__all__
    assert list(inspect.signature(
        discovery.read_plaintext_ocf_resource).parameters) == [
            'host', 'href', 'query', 'port', 'timeout', 'retries', 'family']


def test_dynamic_source_block2_read_preserves_path_query_and_metadata():
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    responder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(('127.0.0.1', 0))
    responder.bind(('127.0.0.1', 0))
    listener.settimeout(2.0)
    responder.settimeout(2.0)
    body = cbor2.dumps({'value': 'x' * 80})
    assert 64 < len(body) <= 128
    errors = []

    def respond():
        try:
            first_request, client = listener.recvfrom(8192)
            first = parse_coap(first_request)
            first_options = _option_map(first[4])
            assert first[:2] == (TYPE_NON, METHOD_GET)
            assert len(first[3]) == 8 and first[5] == b''
            assert first_options[URI_PATH] == [b'oic', b'res']
            assert first_options[URI_QUERY] == [
                b'if=oic.if.baseline', b'rt=oic.wk.d']
            assert first_options[ACCEPT] == [CF_CBOR]
            assert BLOCK2 not in first_options

            common = [
                (CONTENT_FORMAT, CF_CBOR),
                (ETAG, b'fixture'),
                (SIZE2, _uint_bytes(len(body))),
            ]
            responder.sendto(
                build_coap(
                    TYPE_CON,
                    0x45,
                    0x7001,
                    first[3],
                    [*common, (BLOCK2, block_value(0, 1, 2))],
                    body[:64],
                ),
                client,
            )

            acknowledgement, acknowledgement_peer = responder.recvfrom(8192)
            assert acknowledgement_peer == client
            assert parse_coap(acknowledgement)[:4] == (
                TYPE_ACK, 0, 0x7001, b'')

            continuation, continuation_client = responder.recvfrom(8192)
            second = parse_coap(continuation)
            second_options = _option_map(second[4])
            assert continuation_client == client
            assert second[:2] == (TYPE_NON, METHOD_GET)
            assert second[3] == first[3] and second[2] != first[2]
            assert second[5] == b''
            assert second_options[URI_PATH] == [b'oic', b'res']
            assert second_options[URI_QUERY] == [
                b'if=oic.if.baseline', b'rt=oic.wk.d']
            assert second_options[ACCEPT] == [CF_CBOR]
            assert second_options[BLOCK2] == [block_value(1, 0, 2)]

            responder.sendto(
                build_coap(
                    TYPE_NON,
                    0x45,
                    0x7002,
                    first[3],
                    [(BLOCK2, block_value(1, 0, 2))],
                    body[64:],
                ),
                client,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=respond)
    thread.start()
    try:
        result = discovery.read_plaintext_ocf_resource(
            '127.0.0.1',
            '/oic/res',
            query=('if=oic.if.baseline', 'rt=oic.wk.d'),
            port=listener.getsockname()[1],
            timeout=1.5,
            retries=1,
            family=socket.AF_INET,
        )
    finally:
        thread.join(timeout=3.0)
        listener.close()
        responder.close()

    assert not thread.is_alive()
    assert errors == []
    assert result.code == 0x45
    assert result.payload == body
    assert result.blocks_received == 2
    assert result.content_format == int.from_bytes(CF_CBOR, 'big')
    assert result.size2 == len(body)
    assert result.attempts == 2
    assert result.response_received
    assert result.error_code is None
    assert result.complete and result.successful


def test_complete_unauthorized_response_preserves_code_and_body():
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(('127.0.0.1', 0))
    listener.settimeout(2.0)
    body = b'authorization required'
    errors = []

    def respond():
        try:
            request, client = listener.recvfrom(8192)
            token = parse_coap(request)[3]
            listener.sendto(
                build_coap(
                    TYPE_NON,
                    0x81,
                    0x7101,
                    token,
                    [(CONTENT_FORMAT, b'')],
                    body,
                ),
                client,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=respond)
    thread.start()
    try:
        result = discovery.read_plaintext_ocf_resource(
            '127.0.0.1',
            '/temperatures/0',
            port=listener.getsockname()[1],
            timeout=1.0,
            retries=0,
            family=socket.AF_INET,
        )
    finally:
        thread.join(timeout=2.0)
        listener.close()

    assert not thread.is_alive()
    assert errors == []
    assert result.code == 0x81
    assert result.payload == body
    assert result.blocks_received == 1
    assert result.content_format is None
    assert result.size2 is None
    assert result.response_received
    assert result.error_code is None
    assert result.complete and not result.successful


def test_retry_keeps_token_changes_mid_and_stays_in_one_deadline():
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(('127.0.0.1', 0))
    listener.settimeout(2.0)
    errors = []

    def respond():
        try:
            first_request, client = listener.recvfrom(8192)
            second_request, second_client = listener.recvfrom(8192)
            first = parse_coap(first_request)
            second = parse_coap(second_request)
            assert second_client == client
            assert first[:2] == second[:2] == (TYPE_NON, METHOD_GET)
            assert first[3] == second[3]
            assert first[2] != second[2]
            listener.sendto(
                build_coap(
                    TYPE_NON,
                    0x45,
                    0x7201,
                    second[3],
                    [],
                    cbor2.dumps({'value': 1}),
                ),
                client,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=respond)
    thread.start()
    started = time.monotonic()
    try:
        result = discovery.read_plaintext_ocf_resource(
            '127.0.0.1',
            '/oic/d',
            port=listener.getsockname()[1],
            timeout=0.7,
            retries=1,
            family=socket.AF_INET,
        )
    finally:
        elapsed = time.monotonic() - started
        thread.join(timeout=2.0)
        listener.close()

    assert not thread.is_alive()
    assert errors == []
    assert result.successful
    assert result.attempts == 2
    assert elapsed < 1.0


def test_partial_block2_transfer_returns_no_partial_payload():
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(('127.0.0.1', 0))
    listener.settimeout(2.0)
    errors = []

    def respond():
        try:
            request, client = listener.recvfrom(8192)
            token = parse_coap(request)[3]
            listener.sendto(
                build_coap(
                    TYPE_NON,
                    0x45,
                    0x7301,
                    token,
                    [(BLOCK2, block_value(0, 1, 2))],
                    b'x' * 64,
                ),
                client,
            )
            continuation, continuation_client = listener.recvfrom(8192)
            parsed = parse_coap(continuation)
            assert continuation_client == client
            assert parsed[3] == token
            assert _option_map(parsed[4])[BLOCK2] == [
                block_value(1, 0, 2)]
        except Exception as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=respond)
    thread.start()
    started = time.monotonic()
    try:
        result = discovery.read_plaintext_ocf_resource(
            '127.0.0.1',
            '/oic/res',
            port=listener.getsockname()[1],
            timeout=0.4,
            retries=0,
            family=socket.AF_INET,
        )
    finally:
        elapsed = time.monotonic() - started
        thread.join(timeout=2.0)
        listener.close()

    assert not thread.is_alive()
    assert errors == []
    assert result.code is None
    assert result.payload == b''
    assert result.blocks_received == 1
    assert result.response_received
    assert result.error_code == 'malformed_ocf_response'
    assert not result.complete and not result.successful
    assert elapsed < 0.8


def test_wrong_token_confirmable_response_is_ignored_without_ack():
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(('127.0.0.1', 0))
    listener.settimeout(0.4)
    errors = []
    unexpected_datagrams = []

    def respond():
        try:
            request, client = listener.recvfrom(8192)
            token = parse_coap(request)[3]
            listener.sendto(
                build_coap(
                    TYPE_CON,
                    0x45,
                    0x7401,
                    b'badtoken',
                    [],
                    cbor2.dumps({'wrong': True}),
                ),
                client,
            )
            listener.sendto(
                build_coap(
                    TYPE_NON,
                    0x45,
                    0x7402,
                    token,
                    [],
                    cbor2.dumps({'right': True}),
                ),
                client,
            )
            try:
                unexpected_datagrams.append(listener.recvfrom(8192))
            except TimeoutError:
                pass
        except Exception as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=respond)
    thread.start()
    try:
        result = discovery.read_plaintext_ocf_resource(
            '127.0.0.1',
            '/oic/d',
            port=listener.getsockname()[1],
            timeout=0.8,
            retries=0,
            family=socket.AF_INET,
        )
    finally:
        thread.join(timeout=2.0)
        listener.close()

    assert not thread.is_alive()
    assert errors == []
    assert unexpected_datagrams == []
    assert result.successful
    assert cbor2.loads(result.payload) == {'right': True}


def test_resolution_failure_and_result_repr_are_redacted(monkeypatch):
    remote_host = 'private-appliance.invalid'
    resource = '/private/resource'

    def fail(host, port, *, family):
        assert host == remote_host
        assert port == 5683
        assert family == socket.AF_INET6
        raise EndpointError()

    monkeypatch.setattr(discovery, 'resolve_udp_endpoints', fail)

    result = discovery.read_plaintext_ocf_resource(
        remote_host, resource, family=socket.AF_INET6)
    rendered = repr(result) + ''.join(
        traceback.format_exception(EndpointError()))

    assert result.error_code == 'endpoint_unavailable'
    assert result.attempts == 0
    assert remote_host not in rendered
    assert resource not in rendered
    assert 'private body' not in repr(
        discovery.PlaintextOcfResourceResult(
            0x45, b'private body', 1, 60, 12, 1, True))


def test_result_is_immutable():
    result = discovery.PlaintextOcfResourceResult(
        0x45, b'body', 1, 60, 4, 1, True)

    with pytest.raises(FrozenInstanceError):
        result.attempts = 2


@pytest.mark.parametrize(
    ('href', 'kwargs', 'error_type'),
    (
        (None, {}, TypeError),
        (b'/oic/d', {}, TypeError),
        ('oic/d', {}, ValueError),
        ('/', {}, ValueError),
        ('/oic/', {}, ValueError),
        ('/oic//d', {}, ValueError),
        ('/oic/d?x=1', {}, ValueError),
        ('/oic/d#fragment', {}, ValueError),
        ('/' + '/'.join(['x'] * 33), {}, ValueError),
        ('/' + 'x' * 1025, {}, ValueError),
        ('/\ud800', {}, ValueError),
        ('/oic/d', {'query': 'if=oic.if.baseline'}, TypeError),
        ('/oic/d', {'query': ('x=1', b'x=2')}, TypeError),
        ('/oic/d', {'query': ('',)}, ValueError),
        ('/oic/d', {'query': tuple('x' for _ in range(33))}, ValueError),
        ('/oic/d', {'query': ('x' * 1025,)}, ValueError),
        ('/oic/d', {'query': ('\ud800',)}, ValueError),
        ('/oic/d', {'port': 0}, ValueError),
        ('/oic/d', {'port': True}, TypeError),
        ('/oic/d', {'timeout': 0}, ValueError),
        ('/oic/d', {'timeout': float('nan')}, ValueError),
        ('/oic/d', {'timeout': True}, TypeError),
        ('/oic/d', {'retries': 5}, ValueError),
        ('/oic/d', {'retries': True}, TypeError),
        ('/oic/d', {'family': 9999}, ValueError),
        ('/oic/d', {'family': 'AF_INET'}, TypeError),
    ),
)
def test_invalid_inputs_fail_before_network(
        monkeypatch, href, kwargs, error_type):
    def unexpected_resolution(*args, **network_kwargs):
        raise AssertionError((args, network_kwargs))

    monkeypatch.setattr(
        discovery, 'resolve_udp_endpoints', unexpected_resolution)

    with pytest.raises(error_type):
        discovery.read_plaintext_ocf_resource(
            '192.0.2.20', href, **kwargs)
