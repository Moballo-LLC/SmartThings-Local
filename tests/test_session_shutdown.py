"""Two-phase DTLS session shutdown contracts."""

from __future__ import annotations

import threading
import time

import pytest
from OpenSSL import SSL

from smartthings_local.errors import SessionClosedError
from smartthings_local.protocol.dtls_session import DtlsCoapSession


class _NullAuth:
    def configure_context(self, _context):
        return None


class _CloseNotifyConnection:
    def __init__(self):
        self.shutdown_calls = 0
        self._outbound = []

    def shutdown(self):
        self.shutdown_calls += 1
        self._outbound.append(
            b"\x15\xfe\xfd\x00\x00" + b"\x00" * 6
            + b"\x00\x02\x01\x00"
        )

    def bio_read(self, _size):
        if self._outbound:
            return self._outbound.pop(0)
        raise SSL.WantReadError()


class _Socket:
    def __init__(self):
        self.closed = False
        self.sent = []

    def send(self, datagram):
        if self.closed:
            raise OSError("closed")
        self.sent.append(datagram)
        return len(datagram)

    def close(self):
        self.closed = True


def _session(*, rate_limit_rps=1_000_000):
    session = DtlsCoapSession(
        "device.example",
        5684,
        auth=_NullAuth(),
        rate_limit_rps=rate_limit_rps,
    )
    session.conn = _CloseNotifyConnection()
    session.sock = _Socket()
    session.dest = ("192.0.2.10", 5684)
    session.endpoint = object()
    return session


def _run_request(session, operation, started):
    outcome = {}

    def send(_datagram):
        started.set()

    session._send_dgram = send

    def request():
        try:
            if operation == "get":
                session.get(["device", "0"], timeout=30.0)
            elif operation == "post":
                session.post(["mode", "vs", "0"], b"body", timeout=30.0)
            else:
                session.delete(["oic", "sec", "cred"], timeout=30.0)
        except Exception as error:  # noqa: BLE001 - captured for assertion
            outcome["error"] = error

    thread = threading.Thread(target=request)
    thread.start()
    return thread, outcome


@pytest.mark.parametrize("operation", ("get", "post", "delete"))
def test_quiesce_wakes_pending_requests_and_retains_transport(operation):
    session = _session()
    connection = session.conn
    sock = session.sock
    started = threading.Event()
    thread, outcome = _run_request(session, operation, started)

    assert started.wait(1.0)
    session.quiesce_for_close()
    thread.join(1.0)

    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), SessionClosedError)
    assert session.conn is connection
    assert session.sock is sock
    assert session._pending == {}
    assert session._pending_mids == {}
    with pytest.raises(SessionClosedError):
        session._check_live()

    session.close()


def test_quiesce_wakes_paced_observe_without_registering_or_sending():
    session = _session(rate_limit_rps=1.0)
    session._last_send_ts = time.monotonic()
    sends = []
    outcome = {}
    session._send_dgram = sends.append

    def subscribe():
        try:
            session.subscribe(["mode", "vs", "0"])
        except Exception as error:  # noqa: BLE001 - captured for assertion
            outcome["error"] = error

    thread = threading.Thread(target=subscribe)
    thread.start()
    time.sleep(0.02)
    session.quiesce_for_close()
    thread.join(1.0)

    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), SessionClosedError)
    assert session._observe_tokens == {}
    assert sends == []
    session.close()


def test_quiesce_during_observe_send_retires_registered_token():
    session = _session()

    def quiesce_before_send(_datagram):
        session.quiesce_for_close()
        raise SessionClosedError()

    session._send_dgram = quiesce_before_send

    with pytest.raises(SessionClosedError):
        session.subscribe(["mode", "vs", "0"])

    assert session._observe_tokens == {}
    session.close()


def test_quiesce_blocks_direct_application_send():
    session = _session()
    session.quiesce_for_close()

    with pytest.raises(SessionClosedError):
        session._send_dgram(b"request")

    assert session.conn.shutdown_calls == 0
    assert session.sock.sent == []
    session.close()


def test_quiesce_wakes_idle_refetch_worker():
    session = _session()
    thread = threading.Thread(target=session._refetch_loop)
    thread.start()
    time.sleep(0.02)

    session.quiesce_for_close()
    thread.join(0.25)

    assert not thread.is_alive()
    assert session._refetch_pending == {}
    session.close()


def test_close_after_quiesce_flushes_close_notify_without_deregistering():
    session = _session()
    connection = session.conn
    sock = session.sock
    session._observe_tokens = {b"o": "/mode/vs/0"}
    session._send_observe_dereg = lambda *_args: pytest.fail(
        "quiesced close sent an application request"
    )

    session.quiesce_for_close()
    session.close()

    assert connection.shutdown_calls == 1
    assert sock.sent == [
        b"\x15\xfe\xfd\x00\x00" + b"\x00" * 6
        + b"\x00\x02\x01\x00"
    ]
    assert sock.closed
    assert session.sock is None
    assert session.conn is None
    assert session.dest is None
    assert session.endpoint is None
    assert session._observe_tokens == {}

    session.close()
    assert connection.shutdown_calls == 1


def test_abort_closes_without_close_notify_and_is_idempotent():
    session = _session()
    connection = session.conn
    sock = session.sock
    event = threading.Event()
    container = {}
    session._register_pending_request(b"token", event, container)

    session.abort()

    assert connection.shutdown_calls == 0
    assert sock.sent == []
    assert sock.closed
    assert event.is_set()
    assert isinstance(container.get("err"), SessionClosedError)
    assert session.sock is None
    assert session.conn is None
    assert session.dest is None
    assert session.endpoint is None
    assert session._pending == {}
    assert session._pending_mids == {}

    session.abort()


def test_quiesced_session_cannot_start_reader():
    session = _session()
    session.quiesce_for_close()

    with pytest.raises(SessionClosedError):
        session.start_reader()

    assert session._reader_thread is None
    session.close()


def test_normal_close_still_deregisters_before_quiescing(monkeypatch):
    session = _session()
    session._observe_tokens = {b"o": "/mode/vs/0"}
    calls = []
    session._send_observe_dereg = lambda *args: calls.append(args)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    session.close()

    assert calls == [(b"o", ["mode", "vs", "0"])]
