import socket
import time

from smartthings_local.errors import ProbeError
from smartthings_local.protocol import dtls_probe as p


def _rec(content_type, frag):
    """Build one DTLS record: 13-byte header + fragment."""
    return (bytes([content_type])
            + b'\xfe\xfd'                 # DTLS 1.2
            + b'\x00\x00'                 # epoch
            + b'\x00\x00\x00\x00\x00\x00' # sequence number
            + len(frag).to_bytes(2, 'big')
            + frag)


def _hs(msg_type, body=b''):
    return _rec(p._CT_HANDSHAKE, bytes([msg_type]) + body)


def _alert(level, desc):
    return _rec(p._CT_ALERT, bytes([level, desc]))


def test_classify_hello_verify_request():
    assert p.classify_datagram(_hs(3, b'\x00' * 20)) == [
        (p._CT_HANDSHAKE, 'HelloVerifyRequest')]


def test_classify_coalesced_server_flight():
    # OpenSSL commonly hands back ServerHello+Certificate back-to-back.
    dgram = _hs(2, b'\x00' * 30) + _hs(11, b'\x00' * 40)
    assert p.classify_datagram(dgram) == [
        (p._CT_HANDSHAKE, 'ServerHello'),
        (p._CT_HANDSHAKE, 'Certificate')]


def test_classify_fatal_alert_names_description():
    # The OCF-PKI-wall signature: fatal unsupported_certificate (43).
    assert p.classify_datagram(_alert(2, 43)) == [
        (p._CT_ALERT, (2, 'unsupported_certificate'))]


def test_classify_unknown_handshake_type_is_not_lost():
    assert p.classify_datagram(_hs(99)) == [(p._CT_HANDSHAKE, 'hs99')]


def test_dead_port_probe_is_dead_and_never_raises():
    # Nothing listens here; the probe must fold the silence into a DEAD
    # result within the timeout rather than raise.
    r = p.probe('127.0.0.1', 5684, timeout=1.0)
    assert r.outcome == p.DEAD
    assert not r.is_dtls_server
    assert r.datagrams == []


def test_is_dtls_server_reflects_outcome():
    r = p.ProbeResult('h', 1)
    r.outcome = p.LIVE
    assert r.is_dtls_server
    r.outcome = p.REJECTED
    assert r.is_dtls_server
    r.outcome = p.DEAD
    assert not r.is_dtls_server


# --- probe() behavioural tests over a scripted fake UDP socket ----------
#
# OpenSSL runs for real against a memory BIO, so the ClientHello on the
# wire is genuine; only the datagram transport is faked. `responder(fake)`
# is called on every recvfrom and returns the bytes to deliver, or None to
# simulate a lost/silent flight (which sleeps the socket timeout so
# OpenSSL's DTLS retransmit clock advances in real time).

class _FakeSock:
    def __init__(self, responder):
        self._responder = responder
        self._timeout = 0.5
        self.sends = []
        self.recv_calls = 0
        self.closed = False

    def settimeout(self, t):
        self._timeout = t

    def setsockopt(self, *a):
        pass

    def bind(self, *a):
        pass

    def sendto(self, data, dest):
        self.sends.append(data)
        return len(data)

    def recvfrom(self, n):
        self.recv_calls += 1
        resp = self._responder(self)
        if resp is None:
            time.sleep(self._timeout)
            raise socket.timeout()
        return resp, ('127.0.0.1', 5684)

    def close(self):
        self.closed = True


def _patch_sock(monkeypatch, fake):
    monkeypatch.setattr(p.socket, 'socket', lambda *a, **k: fake)


def test_stateless_probe_sends_exactly_one_clienthello(monkeypatch):
    # The §4.2.8 regression guard: a HelloVerifyRequest proves liveness,
    # and the stateless gate must stop there — never emitting the cookie'd
    # second ClientHello that would commit association state on the device.
    fake = _FakeSock(lambda f: _hs(3, b'\x00' * 20))
    _patch_sock(monkeypatch, fake)
    r = p.probe('127.0.0.1', 5684, stateless=True, timeout=2.0)
    assert r.outcome == p.LIVE
    assert len(fake.sends) == 1          # only the initial ClientHello
    assert fake.recv_calls == 1          # stopped on the first flight
    assert fake.closed


def test_retransmit_recovers_from_dropped_first_flight(monkeypatch):
    # The first ClientHello is "lost" (recvfrom times out) until OpenSSL's
    # retransmit timer fires a second flight; only then does the server
    # answer. A single dropped datagram must NOT read as DEAD.
    fake = _FakeSock(lambda f: _hs(3, b'\x00' * 20) if len(f.sends) >= 2
                     else None)
    _patch_sock(monkeypatch, fake)
    r = p.probe('127.0.0.1', 5684, stateless=True, retries=2, timeout=5.0)
    assert r.outcome == p.LIVE
    assert len(fake.sends) == 2          # initial + one retransmit


def test_silent_port_is_dead_only_after_flight_budget(monkeypatch):
    # A truly silent port: DEAD, but only after the initial flight plus
    # `retries` retransmits — not on the first unanswered datagram.
    fake = _FakeSock(lambda f: None)
    _patch_sock(monkeypatch, fake)
    r = p.probe('127.0.0.1', 5684, stateless=True, retries=1, timeout=6.0)
    assert r.outcome == p.DEAD
    assert not r.is_dtls_server
    assert len(fake.sends) == 2          # initial + retries(1) retransmit


def test_diagnostic_mode_feeds_server_flight_back(monkeypatch):
    # The inverse of the stateless guard: stateless=False must NOT stop at
    # the HelloVerifyRequest — it feeds the flight back into OpenSSL to
    # drive the handshake onward (the #16 characterization path). The
    # fed-back record here is a stub, so OpenSSL surfaces an error the
    # moment it processes it, which is precisely what proves the probe did
    # not short-circuit before the write.
    fake = _FakeSock(lambda f: _hs(3, b'\x00' * 20))
    _patch_sock(monkeypatch, fake)
    r = p.probe('127.0.0.1', 5684, stateless=False, timeout=3.0)
    assert r.outcome == p.LIVE           # HVR still proved liveness
    assert isinstance(r.error, ProbeError)  # OpenSSL processed the flight
