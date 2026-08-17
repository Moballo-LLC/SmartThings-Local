"""Deterministic tests for connection-attempt cancellation."""

from __future__ import annotations

import select
import socket
import threading
import time
import traceback

import pytest
from OpenSSL import SSL

from smartthings_local.errors import SessionClosedError, SessionError
from smartthings_local.protocol import dtls_session
from smartthings_local.protocol.dtls_session import (
    ConnectCancellation,
    DtlsCoapSession,
)
from smartthings_local.protocol.endpoint import ResolvedUdpEndpoint


class _Auth:
    def __init__(self, on_configure=None):
        self.on_configure = on_configure

    def configure_context(self, _context):
        if self.on_configure is not None:
            self.on_configure()


class _Connection:
    def __init__(self, *, started=None, on_success=None, succeed=False):
        self.started = started
        self.on_success = on_success
        self.succeed = succeed
        self.bio_writes = []
        self.handshake_calls = 0

    def set_connect_state(self):
        return None

    def set_ciphertext_mtu(self, _mtu):
        return None

    def do_handshake(self):
        self.handshake_calls += 1
        if self.started is not None:
            self.started.set()
        if self.succeed:
            if self.on_success is not None:
                self.on_success()
            return
        raise SSL.WantReadError()

    def bio_read(self, _size):
        raise SSL.WantReadError()

    def bio_write(self, datagram):
        self.bio_writes.append(datagram)
        self.succeed = True

    def DTLSv1_get_timeout(self):
        return None

    def shutdown(self):
        return None


def _session(auth=None):
    return DtlsCoapSession(
        "device.example",
        5684,
        auth=auth or _Auth(),
    )


def _install_connection(monkeypatch, connection, data_socket, *, on_open=None):
    endpoint = ResolvedUdpEndpoint(
        socket.AF_INET,
        ("192.0.2.10", 5684),
    )

    def open_socket(*_args, **_kwargs):
        if on_open is not None:
            on_open()
        return data_socket, endpoint

    monkeypatch.setattr(dtls_session.SSL, "Context", lambda *_args: object())
    monkeypatch.setattr(
        dtls_session.SSL,
        "Connection",
        lambda *_args: connection,
    )
    monkeypatch.setattr(
        dtls_session,
        "open_connected_udp_socket",
        open_socket,
    )
    return endpoint


def _run_connect(session, cancel):
    outcome = {}

    def worker():
        try:
            session.connect(timeout=2.0, cancel=cancel)
        except Exception as error:  # noqa: BLE001 - captured for assertion
            outcome["error"] = error
        else:
            outcome["connected"] = True

    thread = threading.Thread(target=worker)
    thread.start()
    return thread, outcome


@pytest.mark.parametrize("cancel", (True, threading.Event(), object(), "signal"))
def test_connect_cancel_type_is_explicit(cancel):
    with pytest.raises(TypeError, match="ConnectCancellation or None"):
        _session().connect(cancel=cancel)


def test_pre_cancelled_connect_stops_before_context_setup(monkeypatch):
    cancel = ConnectCancellation()
    cancel.set()
    monkeypatch.setattr(
        dtls_session.SSL,
        "Context",
        lambda *_args: pytest.fail("cancelled connect configured TLS"),
    )

    with pytest.raises(SessionClosedError):
        _session().connect(cancel=cancel)


def test_cancel_during_context_setup_stops_before_socket_setup(monkeypatch):
    cancel = ConnectCancellation()
    monkeypatch.setattr(dtls_session.SSL, "Context", lambda *_args: object())
    monkeypatch.setattr(
        dtls_session,
        "open_connected_udp_socket",
        lambda *_args, **_kwargs: pytest.fail(
            "cancelled connect opened a socket"
        ),
    )

    with pytest.raises(SessionClosedError):
        _session(_Auth(cancel.set)).connect(cancel=cancel)


def test_cancel_during_socket_setup_closes_before_handshake(monkeypatch):
    cancel = ConnectCancellation()
    connection = _Connection(succeed=True)
    data_socket, peer = socket.socketpair()
    _install_connection(
        monkeypatch,
        connection,
        data_socket,
        on_open=cancel.set,
    )

    try:
        with pytest.raises(SessionClosedError):
            _session().connect(cancel=cancel)
        assert data_socket.fileno() == -1
        assert connection.handshake_calls == 0
    finally:
        peer.close()


def test_socket_signal_wakes_every_subscribed_waiter():
    cancel = ConnectCancellation()
    first = cancel._subscribe()
    second = cancel._subscribe()
    try:
        cancel.set()
        readable, _, _ = select.select(
            (first[0], second[0]),
            (),
            (),
            0,
        )
        assert set(readable) == {first[0], second[0]}
        assert cancel._unsubscribe(*first)
        assert cancel._unsubscribe(*second)
    finally:
        for reader, writer in (first, second):
            reader.close()
            writer.close()


def test_cancel_wakes_blocked_connect_without_poll_latency(monkeypatch):
    cancel = ConnectCancellation()
    started = threading.Event()
    connection = _Connection(started=started)
    data_socket, peer = socket.socketpair()
    _install_connection(monkeypatch, connection, data_socket)
    session = _session()
    thread, outcome = _run_connect(session, cancel)

    try:
        assert started.wait(1.0)
        before = time.monotonic()
        cancel.set()
        thread.join(1.0)
        elapsed = time.monotonic() - before

        assert not thread.is_alive()
        assert elapsed < 0.25
        assert isinstance(outcome.get("error"), SessionClosedError)
        assert data_socket.fileno() == -1
        assert session.sock is None
        assert session.conn is None
        assert not cancel._writers
    finally:
        cancel.set()
        thread.join(1.0)
        peer.close()


def test_reported_handshake_success_wins_cancel_during_unsubscribe(
    monkeypatch,
):
    class CancelDuringUnsubscribe(ConnectCancellation):
        def _unsubscribe(self, reader, writer):
            self.set()
            return super()._unsubscribe(reader, writer)

    cancel = CancelDuringUnsubscribe()
    connection = _Connection(succeed=True)
    data_socket, peer = socket.socketpair()
    endpoint = _install_connection(monkeypatch, connection, data_socket)
    session = _session()

    try:
        session.connect(cancel=cancel)
        assert cancel.is_set()
        assert session.sock is data_socket
        assert session.conn is connection
        assert session.endpoint is endpoint
        assert session.dest == endpoint.sockaddr
        assert not cancel._writers
    finally:
        session.close()
        peer.close()


def test_cancel_during_backend_failure_does_not_read_unset_completion(
    monkeypatch,
):
    cancel = ConnectCancellation()
    connection = _Connection()

    def fail_after_cancel():
        cancel.set()
        raise SSL.Error("synthetic backend failure")

    connection.do_handshake = fail_after_cancel
    data_socket, peer = socket.socketpair()
    _install_connection(monkeypatch, connection, data_socket)
    session = _session()

    try:
        with pytest.raises(SessionClosedError):
            session.connect(cancel=cancel)
        assert data_socket.fileno() == -1
        assert session.sock is None
        assert session.conn is None
        assert not cancel._writers
    finally:
        peer.close()


def test_successful_connect_does_not_set_cancel_or_close_session(monkeypatch):
    cancel = ConnectCancellation()
    connection = _Connection()
    data_socket, peer = socket.socketpair()
    endpoint = _install_connection(monkeypatch, connection, data_socket)
    peer.send(b"synthetic server flight")
    session = _session()

    try:
        session.connect(cancel=cancel)
        assert not cancel.is_set()
        assert session.sock is data_socket
        assert session.conn is connection
        assert session.endpoint is endpoint
        assert connection.bio_writes == [b"synthetic server flight"]
        assert not cancel._writers
    finally:
        session.close()
        peer.close()


def test_cancellation_socket_failure_is_redacted_and_closes_udp(monkeypatch):
    class FailingCancellation(ConnectCancellation):
        def _subscribe(self):
            raise OSError("credential-value at device.example")

    connection = _Connection()
    data_socket, peer = socket.socketpair()
    _install_connection(monkeypatch, connection, data_socket)

    try:
        with pytest.raises(SessionError) as exc:
            _session().connect(cancel=FailingCancellation())

        formatted = "".join(traceback.format_exception(exc.value))
        assert data_socket.fileno() == -1
        assert exc.value.__context__ is None
        assert "credential-value" not in formatted
        assert "device.example" not in formatted
    finally:
        peer.close()
