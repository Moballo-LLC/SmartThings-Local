"""Session-owned pacing for request sends."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from smartthings_local.errors import SessionClosedError
from smartthings_local.protocol import dtls_session
from smartthings_local.protocol.coap import (
    METHOD_GET,
    METHOD_POST,
    OBSERVE,
    TYPE_ACK,
    TYPE_CON,
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


def test_first_get_post_and_subscribe_are_paced_before_send():
    session = _session()
    order = []
    requests = []

    def pace():
        order.append("pace")

    def send(datagram):
        order.append("send")
        request = parse_coap(datagram)
        requests.append(request)
        _mtype, _code, mid, token, options, _payload = request
        if any(number == OBSERVE for number, _value in options):
            assert session._observe_tokens[token] == "/mode/vs/0"
            return
        session._dispatch_coap(
            build_coap(TYPE_ACK, 0x45, mid, token, [], b"ok")
        )

    session.pace = pace
    session._send_dgram = send

    assert session.get(["device", "0"]) == (0x45, b"ok")
    assert session.post(["mode", "vs", "0"], b"payload") == (0x45, b"ok")
    observe_token = session.subscribe(["mode", "vs", "0"])

    assert session._observe_tokens[observe_token] == "/mode/vs/0"
    assert order == ["pace", "send", "pace", "send", "pace", "send"]
    assert [request[1] for request in requests] == [
        METHOD_GET,
        METHOD_POST,
        METHOD_GET,
    ]


def test_every_subscribe_in_registration_burst_honors_rate_limit(monkeypatch):
    session = _session()
    now = [100.0]
    waits = []
    sends = []

    class StopEvent:
        def wait(self, delay):
            waits.append(delay)
            now[0] += delay

    def send(datagram):
        sends.append(datagram)
        session._last_send_ts = now[0]

    monkeypatch.setattr(dtls_session.time, "monotonic", lambda: now[0])
    session._stop = StopEvent()
    session._min_req_interval = 0.2
    session._last_send_ts = 0.0
    session._send_dgram = send

    for index in range(11):
        session.subscribe(["resource", "vs", str(index)])

    assert len(sends) == 11
    assert waits == [pytest.approx(0.2)] * 10


def test_existing_caller_pacing_before_subscribe_does_not_wait_twice(
    monkeypatch,
):
    session = _session()
    now = [100.05]
    waits = []

    class StopEvent:
        def wait(self, delay):
            waits.append(delay)
            now[0] += delay

    monkeypatch.setattr(dtls_session.time, "monotonic", lambda: now[0])
    session._stop = StopEvent()
    session._min_req_interval = 0.2
    session._last_send_ts = 100.0
    session._send_dgram = Mock()

    session.pace()
    session.subscribe(["mode", "vs", "0"])

    assert waits == [pytest.approx(0.15)]
    session._send_dgram.assert_called_once()


def test_subscribe_rechecks_liveness_after_pacing_before_registering():
    session = _session()
    session._send_dgram = Mock()

    def close_during_pacing():
        session.conn = None

    session.pace = close_during_pacing

    with pytest.raises(SessionClosedError):
        session.subscribe(["mode", "vs", "0"])

    assert session._observe_tokens == {}
    session._send_dgram.assert_not_called()


def test_ack_ping_and_observe_deregister_are_not_paced():
    session = _session()
    session.pace = Mock(side_effect=AssertionError("control send was paced"))

    class Connection:
        def __init__(self):
            self.sent = []

        def send(self, datagram):
            self.sent.append(datagram)

        def bio_read(self, _size):
            return b""

    connection = Connection()
    session.conn = connection

    session.ping()
    session._send_observe_dereg(b"\x40", ["mode", "vs", "0"])
    session._dispatch_coap(
        build_coap(TYPE_CON, 0x45, 0x1234, b"unknown", [], b"state")
    )

    session.pace.assert_not_called()
    assert len(connection.sent) == 3
    assert parse_coap(connection.sent[-1])[:4] == (TYPE_ACK, 0, 0x1234, b"")
