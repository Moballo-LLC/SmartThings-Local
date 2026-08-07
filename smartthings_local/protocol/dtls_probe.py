"""DTLS ClientHello probe — a cheap, deterministic liveness + diagnostic
primitive that sits in front of a full handshake.

Two problems this solves:

  1. Liveness.  A 1-byte UDP probe cannot tell a silent port from a real
     DTLS server: anything that doesn't return ICMP-unreachable looks
     "live", so a discovery loop pays the full HANDSHAKE_TIMEOUT_S
     (12 s) on every false-positive port.  A real DTLS server, by
     contrast, answers a ClientHello with a HelloVerifyRequest (RFC 6347
     §4.2.1 stateless cookie exchange) in ~1 RTT, *before* any
     certificate work.  So one ClientHello round-trip distinguishes the
     real port from dead ones deterministically and cheaply — then the
     expensive cert handshake is committed to exactly one port.

  2. Diagnosis.  The full handshake collapses "no DTLS server here",
     "server up but rejected my cert", and "server up but no shared
     cipher/version" into one opaque timeout/error.  Everything the
     server volunteers about itself — chosen cipher, its cert chain, its
     CertificateRequest, or a fatal Alert — arrives in its first flight,
     *before* we send our own certificate.  Driving the handshake only
     that far (no client cert required) characterizes a device.  This is
     how you tell an OCF-PKI-wall device (rejects at cert-verify) from a
     cipher/version mismatch without a cert it would ever accept.

Reuses split_dtls() (the record framer) and the same memory-BIO pump as
DtlsCoapSession.connect(), so the ClientHello on the wire is byte-for-byte
what our real client emits (same cipher list, same @SECLEVEL=0).
"""

import socket
import time

from OpenSSL import SSL

from ..errors import ProbeError
from .coap import split_dtls
from .dtls_session import _OCF_ROOT_CA, _load_pem_chain

# DTLS record content types (RFC 6347 §4.1)
_CT_CHANGE_CIPHER_SPEC = 20
_CT_ALERT = 21
_CT_HANDSHAKE = 22
_CT_APP_DATA = 23

# Handshake message types (RFC 5246 §7.4 / RFC 6347)
_HS_NAMES = {
    0: 'HelloRequest',
    1: 'ClientHello',
    2: 'ServerHello',
    3: 'HelloVerifyRequest',
    11: 'Certificate',
    12: 'ServerKeyExchange',
    13: 'CertificateRequest',
    14: 'ServerHelloDone',
    15: 'CertificateVerify',
    16: 'ClientKeyExchange',
    20: 'Finished',
}

# TLS alert descriptions (RFC 5246 §7.2) — the ones a picky OCF stack
# actually sends are called out; the rest are here so a probe never
# reports a bare number.
_ALERT_NAMES = {
    0: 'close_notify',
    10: 'unexpected_message',
    20: 'bad_record_mac',
    40: 'handshake_failure',
    42: 'bad_certificate',
    43: 'unsupported_certificate',
    44: 'certificate_revoked',
    45: 'certificate_expired',
    46: 'certificate_unknown',
    47: 'illegal_parameter',
    48: 'unknown_ca',
    49: 'access_denied',
    50: 'decode_error',
    51: 'decrypt_error',
    70: 'protocol_version',
    71: 'insufficient_security',
    80: 'internal_error',
    86: 'inappropriate_fallback',
    90: 'user_canceled',
    112: 'unrecognized_name',
    116: 'certificate_required',
}

# Outcome classes, coarsest first.
DEAD = 'dead'            # no DTLS response at all — silent/non-DTLS port
LIVE = 'live'            # DTLS server confirmed (HelloVerifyRequest/ServerHello)
COMPLETED = 'completed'  # full handshake succeeded (cert accepted)
REJECTED = 'rejected'    # server sent a fatal Alert


class ProbeResult:
    """What a single ClientHello probe learned about one host:port."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.outcome = DEAD
        self.rtt_s = None
        # Ordered, de-duplicated handshake message names the server sent.
        self.handshake_msgs = []
        # (level, description_name) if a fatal/warning Alert was seen.
        self.alert = None
        # Raw inbound datagrams, for callers that want to dig deeper.
        self.datagrams = []
        self.error = None

    @property
    def is_dtls_server(self):
        """True when a DTLS server was proven present, regardless of
        whether it liked our credentials."""
        return self.outcome in (LIVE, COMPLETED, REJECTED)

    def __repr__(self):
        bits = [f'{self.host}:{self.port}', self.outcome]
        if self.rtt_s is not None:
            bits.append(f'{self.rtt_s * 1000:.0f}ms')
        if self.handshake_msgs:
            bits.append('+'.join(self.handshake_msgs))
        if self.alert:
            bits.append(f'alert={self.alert[1]}')
        if self.error:
            bits.append(f'err={self.error}')
        return f'<ProbeResult {" ".join(bits)}>'


def classify_datagram(dgram):
    """Parse one inbound UDP datagram into a list of
    (content_type, detail) tuples — detail is the handshake message name
    for handshake records, an (level, description_name) tuple for alerts,
    or None otherwise.  Pure; safe to unit-test on captured bytes."""
    out = []
    for rec in split_dtls(dgram):
        ct = rec[0]
        frag = rec[13:]
        if ct == _CT_HANDSHAKE and frag:
            out.append((ct, _HS_NAMES.get(frag[0], f'hs{frag[0]}')))
        elif ct == _CT_ALERT and len(frag) >= 2:
            out.append((ct, (frag[0], _ALERT_NAMES.get(frag[1], str(frag[1])))))
        else:
            out.append((ct, None))
    return out


def probe(host, port, *, cert_pem=None, key_pem=None,
          cert_path=None, key_path=None,
          stateless=True, retries=2, timeout=3.0, mtu=1280):
    """Send a DTLS ClientHello to host:port and classify the server's
    first flight.

    Two modes:

      stateless=True (default) — a *liveness* gate.  Stop the instant the
        server proves itself with a HelloVerifyRequest (or ServerHello),
        and never send the cookie'd second ClientHello.  By RFC 6347
        §4.2.1 the server answers the first ClientHello WITHOUT allocating
        association state, so a stateless probe leaves the device
        completely untouched — no orphaned association, no ~8 s §4.2.8
        cooldown for a later real connect from a different source port.
        Outcome is DEAD or LIVE.  This is the mode a discovery/reconnect
        loop should use in front of a real handshake.

      stateless=False — a *diagnostic* drive.  Continue the handshake as
        far as the server's own flight goes (up to ServerHelloDone, or to
        COMPLETED with a client cert), capturing its cipher, cert chain,
        CertificateRequest, or a fatal Alert.  This deliberately commits
        association state on the device, so keep it out of hot reconnect
        paths; it is the tool for characterizing an OCF-PKI-wall device
        (#16) — trust rejection vs cipher/version mismatch.

    A single dropped ClientHello would otherwise read as a false DEAD, so
    the silent path services OpenSSL's DTLS retransmit timer and re-sends
    up to `retries` times before giving up.  A live server still answers
    on the first RTT — retransmit only lengthens the silent path.

    Never raises on a network/handshake failure — those are folded into
    the ProbeResult so a discovery loop can race many ports safely.
    """
    result = ProbeResult(host, port)

    ctx = SSL.Context(SSL.DTLS_METHOD)
    ctx.load_verify_locations(_OCF_ROOT_CA)
    # Accept the chain unconditionally: a probe classifies what the server
    # sends, it does not gate on our trust decision.
    ctx.set_verify(SSL.VERIFY_PEER, lambda *a: True)
    ctx.set_cipher_list(b'ECDHE-ECDSA-AES128-GCM-SHA256:@SECLEVEL=0')
    if cert_pem is not None:
        _load_pem_chain(ctx, cert_pem, key_pem)
    elif cert_path is not None:
        ctx.use_certificate_chain_file(cert_path)
        ctx.use_privatekey_file(key_path)
        ctx.check_privatekey()

    conn = SSL.Connection(ctx, None)
    conn.set_connect_state()
    conn.set_ciphertext_mtu(mtu)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)
    dest = (host, port)

    t0 = time.time()
    seen = set()
    retransmits = 0
    try:
        while time.time() - t0 < timeout:
            try:
                conn.do_handshake()
                result.outcome = COMPLETED
                if result.rtt_s is None:
                    result.rtt_s = time.time() - t0
                break
            except SSL.WantReadError:
                pass
            except SSL.Error:
                # A fatal Alert lands here; the alert record was already
                # captured below, so classification still works.
                result.error = ProbeError()
                break

            try:
                o = conn.bio_read(65535)
                if o:
                    for r in split_dtls(o):
                        sock.sendto(r, dest)
            except SSL.WantReadError:
                pass

            try:
                d, _ = sock.recvfrom(65535)
            except socket.timeout:
                # No answer to the last flight. Service OpenSSL's DTLS
                # retransmit timer: once it has counted down to 0,
                # handle_timeout() re-queues the previous flight into the
                # write BIO for the next iteration to flush. A live server
                # answers within a flight or two; a silent/non-DTLS port
                # never does, so we give up only after `retries`
                # retransmits — one dropped ClientHello no longer reads as
                # a false DEAD.
                to = conn.DTLSv1_get_timeout()
                if to is not None and to <= 0:
                    if retransmits >= retries:
                        break
                    conn.DTLSv1_handle_timeout()
                    retransmits += 1
                continue
            if not d:
                continue

            if result.rtt_s is None:
                result.rtt_s = time.time() - t0
            result.datagrams.append(d)
            server_flight = False
            for ct, detail in classify_datagram(d):
                if ct == _CT_HANDSHAKE:
                    if detail not in seen:
                        seen.add(detail)
                        result.handshake_msgs.append(detail)
                    if result.outcome == DEAD:
                        result.outcome = LIVE
                    if detail in ('HelloVerifyRequest', 'ServerHello'):
                        server_flight = True
                elif ct == _CT_ALERT and detail is not None:
                    level, name = detail
                    result.alert = (level, name)
                    if level == 2:  # fatal
                        result.outcome = REJECTED
            # Stateless liveness: the server proved itself with a
            # HelloVerifyRequest/ServerHello, which it answered without
            # allocating state. Stop before feeding this flight back to
            # OpenSSL — doing so would make it emit the cookie'd second
            # ClientHello, the message that actually commits association
            # state on the device. Not writing it keeps the probe
            # zero-footprint.
            if stateless and server_flight:
                break
            conn.bio_write(d)
    finally:
        sock.close()

    return result


def _main(argv):
    import concurrent.futures as cf

    if len(argv) < 2:
        print('usage: python -m smartthings_local.protocol.dtls_probe '
              'HOST PORT [PORT...] [--cert FILE --key FILE] [--stateless]')
        return 2
    host = argv[0]
    cert_path = key_path = None
    # CLI defaults to the diagnostic drive so `HOST PORT` characterizes a
    # device (cipher/cert/Alert). Pass --stateless for the zero-footprint
    # liveness gate a reconnect loop would use.
    stateless = False
    ports = []
    it = iter(argv[1:])
    for a in it:
        if a == '--cert':
            cert_path = next(it)
        elif a == '--key':
            key_path = next(it)
        elif a == '--stateless':
            stateless = True
        else:
            ports.append(int(a))

    # Race the ports: a ClientHello probe is cheap, so fan out and let the
    # live one answer in ~1 RTT instead of serializing 12 s timeouts.
    with cf.ThreadPoolExecutor(max_workers=max(1, len(ports))) as ex:
        futs = {ex.submit(probe, host, p, cert_path=cert_path,
                          key_path=key_path, stateless=stateless): p
                for p in ports}
        results = [f.result() for f in cf.as_completed(futs)]

    for r in sorted(results, key=lambda r: r.port):
        print(r)
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(_main(sys.argv[1:]))
