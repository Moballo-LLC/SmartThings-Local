"""Reader-thread death visibility (QuiteYellow/SmartThings-Local#37).

A connected UDP socket surfaces ICMP errors on recv; before this the
reader exited silently on the first one and every later request waited
out its full timeout against a session nobody was reading. These tests
pin the three behaviours that fixed it: advisory ICMP errnos keep the
reader alive, a real socket error exits with a WARNING and clears
_reader_running, and callers then fail fast with SessionClosedError.
"""
import errno
import logging
import socket
import threading
import time

import pytest
from OpenSSL import SSL

from smartthings_local.errors import SessionClosedError
from smartthings_local.protocol.dtls_session import DtlsCoapSession

_LOGGER_NAME = "smartthings_local.protocol.dtls_session"


class _NullAuth:
    """Structural AuthenticationProvider — never configured, we skip connect()."""

    def configure_context(self, _context):
        return None


class _FakeConn:
    """Minimal stand-in for SSL.Connection: each datagram written to the
    BIO surfaces as one decrypted packet on the next recv(), then
    WantReadError like a drained DTLS record buffer."""

    def __init__(self):
        self._decrypted = []

    def bio_write(self, datagram):
        self._decrypted.append(datagram)

    def recv(self, _n):
        if self._decrypted:
            return self._decrypted.pop(0)
        raise SSL.WantReadError()

    def bio_read(self, _n):
        return b""

    def send(self, _datagram):
        return None

    def shutdown(self):
        return None


class _FakeSock:
    """Scripted UDP socket. Each step is bytes to return, an exception to
    raise, or a callable to run (then a timeout, so the loop re-checks
    _stop). An exhausted script blocks like a real recv timeout; once
    close()d it raises EBADF the way a closed fd does."""

    def __init__(self, steps=()):
        self._steps = list(steps)
        self.closed = False
        self.timeout = None

    def settimeout(self, value):
        self.timeout = value

    def recv(self, _n):
        if self.closed:
            raise OSError(errno.EBADF, "bad file descriptor")
        if self._steps:
            step = self._steps.pop(0)
            if callable(step):
                step()
                raise socket.timeout()
            if isinstance(step, BaseException):
                raise step
            return step
        time.sleep(0.01)
        raise socket.timeout()

    def send(self, data):
        return len(data)

    def close(self):
        self.closed = True


def _make_session():
    sess = DtlsCoapSession("host", 1234, auth=_NullAuth())
    sess.conn = _FakeConn()
    return sess


def _run_reader(sess, steps, timeout=2.0):
    sess.sock = _FakeSock(steps)
    sess.start_reader()
    sess._reader_thread.join(timeout)
    assert not sess._reader_thread.is_alive(), "reader thread did not exit"


def test_advisory_icmp_error_does_not_kill_reader(caplog):
    sess = _make_session()
    dispatched = []
    sess._dispatch_coap = dispatched.append

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        _run_reader(sess, [
            OSError(errno.ECONNREFUSED, "connection refused"),
            b"\x60\x00\x00\x00",              # survives, gets dispatched
            lambda: sess._stop.set(),         # end the loop cleanly
        ])

    assert dispatched == [b"\x60\x00\x00\x00"]
    assert not sess._reader_running.is_set()
    assert any(r.levelno == logging.DEBUG and "advisory" in r.getMessage()
               for r in caplog.records)
    # An advisory errno is not a real exit — no WARNING.
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_fatal_socket_error_exits_with_warning(caplog):
    sess = _make_session()

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _run_reader(sess, [OSError(errno.EBADF, "bad file descriptor")])

    assert not sess._reader_running.is_set()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "reader exiting" in warnings[0].getMessage()


def test_fatal_reader_exit_drains_mid_registry_and_wakes_waiters():
    sess = _make_session()
    pending = []
    for token in (b"get", b"post"):
        event = threading.Event()
        container = {}
        sess._register_pending_request(token, event, container)
        pending.append((event, container))

    _run_reader(sess, [OSError(errno.EBADF, "bad file descriptor")])

    assert sess._pending == {}
    assert sess._pending_mids == {}
    assert all(event.is_set() for event, _container in pending)
    assert all(
        isinstance(container.get("err"), SessionClosedError)
        for _event, container in pending
    )


def test_request_fails_fast_after_reader_death():
    sess = _make_session()
    _run_reader(sess, [OSError(errno.EBADF, "bad file descriptor")])
    assert not sess._reader_running.is_set()

    start = time.monotonic()
    with pytest.raises(SessionClosedError):
        sess.get(["oic", "d"], timeout=10.0)
    elapsed = time.monotonic() - start
    # The whole point: no waiting out the request timeout.
    assert elapsed < 1.0, f"get() waited {elapsed:.2f}s instead of failing fast"


def test_close_does_not_log_warning_on_teardown(caplog):
    sess = _make_session()
    sess.sock = _FakeSock()          # empty script: blocks on recv
    sess.start_reader()
    time.sleep(0.05)                 # let the reader reach recv

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        sess.close()
        sess._reader_thread.join(2.0)

    assert not sess._reader_thread.is_alive()
    assert not sess._reader_running.is_set()
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_check_live_without_reader_matches_old_conn_guard():
    sess = _make_session()          # conn set, reader never started
    assert sess._reader_thread is None
    sess._check_live()              # must not raise — config-flow behaviour

    sess.conn = None
    with pytest.raises(SessionClosedError):
        sess._check_live()
