"""Shared MID lifecycle for token-correlated CoAP requests."""

from __future__ import annotations

import threading

import pytest

from smartthings_local.errors import (
    EndpointError,
    SessionClosedError,
    SessionError,
    SessionResetError,
    SessionTimeoutError,
)
from smartthings_local.protocol import dtls_session
from smartthings_local.protocol.coap import (
    TYPE_ACK,
    TYPE_CON,
    TYPE_NON,
    TYPE_RST,
    build_coap,
    parse_coap,
)
from smartthings_local.protocol.dtls_session import DtlsCoapSession


class _NullAuth:
    def configure_context(self, _context):
        return None


def _session():
    session = DtlsCoapSession(
        "device.example",
        5684,
        auth=_NullAuth(),
        rate_limit_rps=1_000_000,
    )
    session.conn = object()
    return session


def _request(session, operation, *, timeout=1.0):
    if operation == "get":
        return session.get(["device", "0"], timeout=timeout)
    return session.post(["mode", "vs", "0"], b"payload", timeout=timeout)


@pytest.mark.parametrize("operation", ["get", "post"])
def test_request_is_indexed_by_token_and_mid_before_send(operation):
    session = _session()

    def send(datagram):
        mtype, _code, mid, token, _options, _payload = parse_coap(datagram)
        assert mtype == TYPE_CON
        exchange = session._pending_mids[mid]
        assert session._pending[token] is exchange.pending
        session._dispatch_coap(
            build_coap(TYPE_ACK, 0x45, mid, token, [], b"ok")
        )

    session._send_dgram = send

    assert _request(session, operation) == (0x45, b"ok")
    assert session._pending == {}
    assert session._pending_mids == {}


@pytest.mark.parametrize("operation", ["get", "post"])
def test_empty_ack_keeps_request_pending_for_separate_response(operation):
    session = _session()

    def send(datagram):
        mtype, code, mid, token, _options, _payload = parse_coap(datagram)
        if mtype == TYPE_ACK and code == 0:
            return

        exchange = session._pending_mids[mid]
        session._dispatch_coap(
            build_coap(TYPE_ACK, 0, mid, b"", [])
        )

        assert exchange.acknowledged is True
        assert session._pending_mids[mid] is exchange
        assert session._pending[token] is exchange.pending
        assert exchange.pending[1] == {}

        session._dispatch_coap(
            build_coap(TYPE_CON, 0x45, mid + 1, token, [], b"ok")
        )

    session._send_dgram = send

    assert _request(session, operation) == (0x45, b"ok")
    assert session._pending == {}
    assert session._pending_mids == {}


def test_empty_ack_stops_get_retransmission(monkeypatch):
    session = _session()
    requests = []
    waits = []

    monkeypatch.setattr(dtls_session, "_BLOCK_ACK_TIMEOUT", 0.01)

    def send(datagram):
        mtype, code, mid, token, _options, _payload = parse_coap(datagram)
        if mtype == TYPE_ACK and code == 0:
            return
        requests.append((mid, token))
        session._dispatch_coap(build_coap(TYPE_ACK, 0, mid, b"", []))

    def wait_for_block(_event, timeout):
        waits.append(timeout)
        mid, token = requests[-1]
        session._dispatch_coap(
            build_coap(TYPE_CON, 0x45, mid + 1, token, [], b"ok")
        )
        return True

    session._send_dgram = send
    session._wait_for_block = wait_for_block

    assert session.get(["device", "0"], timeout=1.0) == (0x45, b"ok")
    assert len(requests) == 1
    assert waits and waits[0] > dtls_session._BLOCK_ACK_TIMEOUT


@pytest.mark.parametrize("operation", ["get", "post"])
def test_matching_bare_rst_fails_request(operation):
    session = _session()

    def send(datagram):
        _mtype, _code, mid, _token, _options, _payload = parse_coap(datagram)
        session._dispatch_coap(build_coap(TYPE_RST, 0, mid, b"", []))

    session._send_dgram = send

    with pytest.raises(SessionError) as raised:
        _request(session, operation)

    # A RST refuses one exchange while the transport stays up, so it is its
    # own type rather than the generic session failure.
    assert type(raised.value) is SessionResetError
    assert session._pending == {}
    assert session._pending_mids == {}


@pytest.mark.parametrize("mtype", [TYPE_ACK, TYPE_RST])
def test_unknown_bare_ack_or_rst_is_ignored(mtype):
    session = _session()
    event = threading.Event()
    container = {}
    mid, exchange = session._register_pending_request(
        b"token", event, container
    )
    try:
        session._dispatch_coap(
            build_coap(mtype, 0, (mid + 1) & 0xFFFF, b"", [])
        )

        assert not event.is_set()
        assert exchange.acknowledged is False
        assert container == {}
    finally:
        session._unregister_pending_request(b"token", mid, exchange)


@pytest.mark.parametrize("mtype", [TYPE_ACK, TYPE_RST])
@pytest.mark.parametrize(
    "malformed",
    ["token", "wrong_version", "empty_payload_marker"],
)
def test_malformed_empty_control_message_is_ignored(mtype, malformed):
    session = _session()
    event = threading.Event()
    container = {}
    mid, exchange = session._register_pending_request(
        b"token", event, container
    )
    try:
        if malformed == "token":
            datagram = build_coap(mtype, 0, mid, b"token", [])
        elif malformed == "wrong_version":
            datagram = bytes([
                (2 << 6) | (mtype << 4),
                0,
                mid >> 8,
                mid & 0xFF,
            ])
        else:
            datagram = build_coap(mtype, 0, mid, b"", []) + b"\xFF"
        session._dispatch_coap(datagram)

        assert not event.is_set()
        assert exchange.acknowledged is False
        assert container == {}
    finally:
        session._unregister_pending_request(b"token", mid, exchange)


@pytest.mark.parametrize("operation", ["get", "post"])
@pytest.mark.parametrize("outcome", ["send_failure", "timeout"])
def test_failure_paths_unregister_both_indices(operation, outcome):
    session = _session()

    if outcome == "send_failure":
        def fail_send(_datagram):
            raise EndpointError()

        session._send_dgram = fail_send
        expected = EndpointError
        timeout = 1.0
    else:
        session._send_dgram = lambda _datagram: None
        if operation == "get":
            session._wait_for_block = lambda _event, _timeout: False
        expected = SessionTimeoutError
        timeout = 0.0

    with pytest.raises(expected):
        _request(session, operation, timeout=timeout)

    assert session._pending == {}
    assert session._pending_mids == {}


def test_close_drains_both_indices_and_wakes_waiters():
    session = _session()
    pending = []
    for token in (b"get", b"post"):
        event = threading.Event()
        container = {}
        session._register_pending_request(token, event, container)
        pending.append((event, container))

    session.close()

    assert session._pending == {}
    assert session._pending_mids == {}
    assert all(event.is_set() for event, _container in pending)
    assert all(
        isinstance(container.get("err"), SessionClosedError)
        for _event, container in pending
    )


@pytest.mark.parametrize("operation", ["get", "post"])
def test_request_cannot_register_after_close_drains(operation):
    session = _session()
    sends = []
    register_entered = threading.Event()
    resume_registration = threading.Event()
    drained = threading.Event()
    release_close = threading.Event()
    original_register = session._register_pending_request
    original_drain = session._close_pending_requests
    outcome = {}
    session._send_dgram = sends.append

    def pause_before_registration(token, event, container):
        register_entered.set()
        resume_registration.wait(1.0)
        return original_register(token, event, container)

    def pause_after_drain():
        original_drain()
        drained.set()
        release_close.wait(1.0)

    def request():
        try:
            _request(session, operation, timeout=0.0)
        except Exception as error:
            outcome["error"] = error

    session._register_pending_request = pause_before_registration
    session._close_pending_requests = pause_after_drain
    request_thread = threading.Thread(target=request)
    close_thread = threading.Thread(target=session.close)
    request_thread.start()
    assert register_entered.wait(1.0)
    close_thread.start()
    assert drained.wait(1.0)
    try:
        resume_registration.set()
        request_thread.join(1.0)
        assert not request_thread.is_alive()
        assert isinstance(outcome.get("error"), SessionClosedError)
        assert sends == []
        assert session._pending == {}
        assert session._pending_mids == {}
    finally:
        resume_registration.set()
        release_close.set()
        request_thread.join(1.0)
        close_thread.join(1.0)
    assert not request_thread.is_alive()
    assert not close_thread.is_alive()


def test_mid_allocation_skips_a_live_exchange_across_wrap():
    session = _session()
    session._mid = 0xFFFF
    first_mid, first = session._register_pending_request(
        b"first", threading.Event(), {}
    )
    session._mid = 0xFFFF
    second_mid, second = session._register_pending_request(
        b"second", threading.Event(), {}
    )
    try:
        assert first_mid == 0
        assert second_mid == 1
        assert session._pending_mids[first_mid] is first
        assert session._pending_mids[second_mid] is second
    finally:
        session._unregister_pending_request(b"first", first_mid, first)
        session._unregister_pending_request(b"second", second_mid, second)


def test_non_control_response_still_resolves_by_token():
    session = _session()
    event = threading.Event()
    container = {}
    request_mid, exchange = session._register_pending_request(
        b"token", event, container
    )
    try:
        session._dispatch_coap(
            build_coap(TYPE_NON, 0x45, 0x1234, b"token", [], b"ok")
        )

        assert event.is_set()
        assert container["code"] == 0x45
        assert container["mid"] == 0x1234
        assert container["payload"] == b"ok"
    finally:
        session._unregister_pending_request(
            b"token", request_mid, exchange
        )
