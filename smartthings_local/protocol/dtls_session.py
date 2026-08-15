"""CoAP-over-DTLS client for Samsung RT-OCF appliances (RFC 7252 + 6347).

Replaces the TLS-over-TCP transport used in the original dryer bridge.
Both the oven (UDP/49154) and the dryer (UDP/49155) speak CoAP-over-DTLS
with the ECDHE-ECDSA-AES128-GCM-SHA256 cipher and a client cert.

Wire-level details that matter (from local-tools/oven-findings.md §17):
  * DTLS ciphertext MTU must be 1200; otherwise OpenSSL fragments the
    client cert across two datagrams and TizenRT drops the second.
  * Samsung's RT-OCF uses ACK+separate-CON for the larger responses.
    The reader MUST correlate by (token, mid) — not arrival order —
    or interleaved one-shot / OBSERVE traffic mis-attributes.
  * Multi-block GET requires the SAME CoAP token across every block
    of the response ("token-stable Block2"). Fresh-token-per-block
    is silently dropped by the server.

Reader thread owns the UDP socket. Callers issue get()/post() and block
on a per-token Event the reader signals. OBSERVE notifications are
delivered via the on_notification callback.
"""
import errno
import math
import os
import socket
import threading
import time

from OpenSSL import SSL

from ..errors import (
    BlockwiseError,
    EndpointError,
    SessionClosedError,
    SessionError,
    SessionTimeoutError,
)
from .coap import (
    URI_PATH, URI_QUERY, OBSERVE, CONTENT_FORMAT, ACCEPT, BLOCK2, SIZE2,
    TYPE_CON, TYPE_NON, TYPE_ACK, TYPE_RST,
    METHOD_GET, METHOD_POST, CF_CBOR,
    OBSERVE_REGISTER, OBSERVE_DEREGISTER, BLOCK_SZX,
    encode_options, parse_coap, build_coap, block_value, fmt_code,
    split_dtls as _split_dtls,
)
from .auth import (
    AuthenticationProvider,
    CertificateAuth,
    _DTLS_CIPHERS,
    _OCF_ROOT_CA,
    _load_pem_chain,
)
from .dtls_handshake import _HANDSHAKE_POLL_S, _drive_dtls_handshake
from .endpoint import open_connected_udp_socket
import logging

logger = logging.getLogger(__name__)

# Diagnostic logging — when DEBUG_BRIDGE=1 in env, the bridge dumps
# every received CoAP frame, every /operational/state/vs/0 + /oven/vs/0
# + /power/vs/0 + /mode/vs/0-options rep change, the full link tree at
# seed time, and the /oic/res directory. Useful for reverse-engineering
# new resources and field semantics; otherwise quiet.
DEBUG_BRIDGE = os.environ.get('DEBUG_BRIDGE') == '1'

# Per-block retransmission: send up to this many times before giving up.
# Each attempt waits at most _BLOCK_ACK_TIMEOUT seconds (capped by the
# overall deadline). Matches RFC 7252 CON retransmit behaviour.
_BLOCK_MAX_ATTEMPTS = 3
_BLOCK_ACK_TIMEOUT  = 4.0

# Inter-request pacing: minimum seconds between CoAP CON sends on one session.
# Samsung's RT-OCF stacks drop requests when hit faster than their firmware
# ceiling (dryer ~14 req/s, oven ~8 req/s, dishwasher unknown). 5 req/s
# (200 ms) is conservative enough for all tested devices; tune per device
# once the ceiling is measured empirically.
_DEFAULT_RATE_LIMIT_RPS = 5.0

# ICMP errors a connected UDP socket surfaces on the next recv. On these
# appliances they show up while the device is rebooting, while it holds an
# orphaned association, or across a router blip, and the next datagram
# usually works. UDP delivery was never guaranteed, so treat them as
# advisory and keep reading. Unconnected sockets never see any of this,
# which is why the reader survived them before the connected-socket change
# in d677c72 (v0.1.3).
_ADVISORY_ERRNOS = frozenset(
    value for value in (
        getattr(errno, name, None)
        for name in ('ECONNREFUSED', 'EHOSTUNREACH', 'ENETUNREACH',
                     'EHOSTDOWN', 'ENETDOWN')
    ) if value is not None
)

def _validate_handshake_timeout(timeout, default):
    """Return one finite, positive DTLS handshake timeout."""
    value = default if timeout is None else timeout
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('timeout must be a number or None')
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(
            'timeout must be a positive finite number or None') from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError('timeout must be a positive finite number or None')
    return value

class DtlsCoapSession:
    """Single sustained DTLS-CoAP session.

    Caller drives lifecycle:
        sess = DtlsCoapSession(host, port, cert_path=cert, key_path=key)
        sess.connect()
        sess.start_reader()
        sess.subscribe([...], on_notification=cb)   # OBSERVE
        code, body = sess.get(['device', '0'])      # Block2 fetch
        code, _    = sess.post(['mode','vs','0'], cbor)
        sess.close()

    Authentication comes from an immutable provider. For compatibility,
    cert_path/key_path and cert_pem/key_pem create a CertificateAuth provider
    internally — exactly one legacy pair is required when auth is omitted.
    """

    HANDSHAKE_TIMEOUT_S = 12.0
    READER_RECV_TIMEOUT_S = 1.0  # short so stop_event propagates quickly
    MAX_BLOCKS = 32              # safety bound for Block2 fetches

    def __init__(self, host, port, cert_path=None, key_path=None, *,
                 cert_pem=None, key_pem=None,
                 on_notification=None, mtu=1200,
                 rate_limit_rps: float = _DEFAULT_RATE_LIMIT_RPS,
                 local_port=None, family=socket.AF_UNSPEC,
                 auth: AuthenticationProvider | None = None):
        file_supplied = cert_path is not None or key_path is not None
        memory_supplied = cert_pem is not None or key_pem is not None
        if auth is not None and (file_supplied or memory_supplied):
            raise ValueError(
                "pass auth or legacy certificate arguments, not both")
        if auth is None and file_supplied and memory_supplied:
            raise ValueError(
                "pass either cert_path/key_path or cert_pem/key_pem, not both")
        if auth is None and memory_supplied:
            if cert_pem is None or key_pem is None:
                raise ValueError("cert_pem and key_pem must be passed together")
        elif auth is None and (cert_path is None or key_path is None):
            raise ValueError(
                "must pass either cert_path/key_path or cert_pem/key_pem")
        if auth is not None and not isinstance(auth, AuthenticationProvider):
            raise TypeError("auth must implement AuthenticationProvider")

        self.host = host
        self.port = port
        self.cert_path = str(cert_path) if cert_path is not None else None
        self.key_path  = str(key_path) if key_path is not None else None
        self.cert_pem = cert_pem
        self.key_pem  = key_pem
        if auth is None:
            if cert_pem is not None:
                auth = CertificateAuth.from_memory(cert_pem, key_pem)
            else:
                auth = CertificateAuth.from_files(self.cert_path, self.key_path)
        self.auth = auth
        self.on_notification = on_notification  # fn(href, payload_bytes)
        self.mtu = mtu
        self._min_req_interval = 1.0 / rate_limit_rps
        # Optional fixed UDP source port. A client that dies without
        # close_notify leaves an orphaned DTLS association on the device,
        # keyed to the old 5-tuple; reconnecting from a fresh ephemeral
        # port presents as a *new* peer and the orphan lingers until the
        # device's own timer reaps it (observed 5-15 min on always-on
        # appliances). Binding the same source port on every connect makes
        # a restart re-handshake over the SAME 5-tuple, which RFC 6347
        # §4.2.8 requires the server to treat as a rebooted peer: complete
        # the new handshake and discard the old association. Verified
        # accepted by RT-OCF (oven, 2026-07-26).
        self.local_port = local_port
        self.family = family

        self.sock = None
        self.conn = None
        self.dest = None
        self.endpoint = None

        self._send_lock = threading.Lock()
        # Randomize MID and token counter starting points so reconnects
        # don't reuse identifiers from previous sessions — Samsung's
        # RT-OCF appears to remember observer state across DTLS
        # sessions, and re-registering with a token it still thinks is
        # active is silently no-ops.
        self._mid = int.from_bytes(os.urandom(2), 'big')
        self._tok_counter = int.from_bytes(os.urandom(4), 'big')
        # OBSERVE tokens are 1-byte (Samsung silently drops TKL>1
        # OBSERVE registrations). Pick a random starting byte in the
        # 0x40..0xff range so each session uses fresh values.
        self._observe_tok_counter = 0x40 + (os.urandom(1)[0] & 0xBF)
        # token (bytes) → (Event, container_dict)
        self._pending = {}
        # token (bytes) → href (str)
        self._observe_tokens = {}

        self._stop = threading.Event()
        self._reader_thread = None
        # Set while the reader owns the socket. Cleared when it exits for
        # any reason, so callers fail fast through _check_live() instead of
        # waiting out a request timeout against a session nobody is reading.
        self._reader_running = threading.Event()
        self._last_send_ts = 0.0

    def pace(self) -> None:
        """Sleep only the part of the rate-limit interval not already consumed
        since the last real send. Uses _stop so session teardown wakes it."""
        remaining = self._min_req_interval - (time.monotonic() - self._last_send_ts)
        if remaining > 0:
            self._stop.wait(remaining)

    # ---- lifecycle ---------------------------------------------------

    def connect(self, *, timeout: float | None = None):
        """Perform a DTLS handshake within a monotonic deadline.

        ``timeout`` overrides ``HANDSHAKE_TIMEOUT_S`` for this call. OpenSSL
        owns DTLS retransmission timing while every receive is capped by the
        remaining budget, so wall-clock adjustments cannot change the bound.
        """
        handshake_timeout = _validate_handshake_timeout(
            timeout, self.HANDSHAKE_TIMEOUT_S)
        deadline = time.monotonic() + handshake_timeout
        ctx = SSL.Context(SSL.DTLS_METHOD)
        self.auth.configure_context(ctx)

        conn = SSL.Connection(ctx, None)
        conn.set_connect_state()
        conn.set_ciphertext_mtu(self.mtu)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SessionTimeoutError()
        sock, endpoint = open_connected_udp_socket(
            self.host,
            self.port,
            family=self.family,
            local_port=self.local_port,
            timeout=min(_HANDSHAKE_POLL_S, remaining),
        )
        dest = endpoint.sockaddr

        backend_failed = False
        io_failed = False
        try:
            completed = _drive_dtls_handshake(
                conn,
                sock,
                deadline=deadline,
            )
        except SSL.Error:
            backend_failed = True
        except OSError:
            io_failed = True
        if backend_failed:
            sock.close()
            raise SessionError() from ConnectionError('DTLS backend failed')
        if io_failed:
            sock.close()
            raise EndpointError() from OSError('UDP handshake I/O failed')
        if not completed:
            sock.close()
            raise SessionTimeoutError()

        self.sock = sock
        self.conn = conn
        self.dest = dest
        self.endpoint = endpoint
        self._stop.clear()

    def start_reader(self):
        """Spawn the reader thread. Must be called after connect()."""
        if self.sock is None:
            raise RuntimeError("connect() before start_reader()")
        self._reader_running.set()
        t = threading.Thread(target=self._reader_loop,
                             daemon=True, name='dtls-reader')
        t.start()
        self._reader_thread = t

    def _check_live(self):
        """Raise if the session cannot carry a request. A dead reader is
        as fatal as a closed connection: the socket may still accept
        sends, but no response will ever be dispatched, so waiting out the
        request timeout only delays the inevitable SessionClosedError.

        Callers that never start a reader (config-flow style) keep the old
        behaviour — only the conn check applies while _reader_thread is
        None."""
        if self.conn is None:
            raise SessionClosedError()
        if self._reader_thread is not None and \
                not self._reader_running.is_set():
            raise SessionClosedError()

    def join(self):
        """Block until the reader thread exits (i.e. socket dies)."""
        if self._reader_thread is not None:
            self._reader_thread.join()

    def _send_observe_dereg(self, tok, path_segs):
        """Send a single OBSERVE deregister GET (Observe option = 1)
        on the existing token. Best-effort — caller swallows errors."""
        if self.conn is None:
            return
        mid = self._next_mid()
        opts = [(URI_PATH, s.encode()) for s in path_segs]
        opts.append((OBSERVE, OBSERVE_DEREGISTER))
        opts.append((ACCEPT, CF_CBOR))
        self._send_dgram(
            build_coap(TYPE_CON, METHOD_GET, mid, tok, opts))

    def close(self):
        """Tear down session. Sends best-effort OBSERVE deregisters
        first so Samsung's RT-OCF cleans up its observer table —
        without this, the per-cert observer state survives DTLS close
        and a quick reconnect with the same tokens silently no-ops."""
        # Send dereg for every active observation while the conn is
        # still healthy. Tiny sleep lets the records reach the wire
        # before we shut DTLS down.
        if self.conn is not None and self._observe_tokens:
            for tok, href in list(self._observe_tokens.items()):
                segs = [s for s in href.split('/') if s]
                try:
                    self._send_observe_dereg(tok, segs)
                except Exception as e:
                    logger.warning("dereg %s: %s", href, e)
            time.sleep(0.1)

        self._stop.set()
        if self.conn is not None:
            try:
                self.conn.shutdown()
            except Exception:
                pass
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        for tok, (ev, container) in list(self._pending.items()):
            container.setdefault('err', SessionClosedError())
            ev.set()
        self._pending.clear()
        self._observe_tokens.clear()
        self.sock = None
        self.conn = None
        self.dest = None
        self.endpoint = None

    # ---- send / receive plumbing -------------------------------------

    def _next_mid(self):
        self._mid = (self._mid + 1) & 0xFFFF
        return self._mid

    def _next_tok(self):
        self._tok_counter = (self._tok_counter + 1) & 0xFFFFFFFF
        # 4-byte tokens — fits within tkl=8 cap with headroom and
        # avoids collisions across long-running OBSERVE subscriptions.
        return self._tok_counter.to_bytes(4, 'big')

    def _next_observe_tok(self):
        # Single-byte tokens for OBSERVE registrations. Samsung
        # RT-OCF accepts these but silently drops TKL=4 OBSERVE
        # registrations. Counter is randomly seeded per session so
        # reconnects don't collide with stale observer state Samsung
        # may still be holding from the previous run.
        self._observe_tok_counter = (self._observe_tok_counter + 1) & 0xFF
        # Avoid 0x00 — some CoAP stacks treat an all-zero token as
        # equivalent to "no token" / empty (TKL=0).
        if self._observe_tok_counter == 0:
            self._observe_tok_counter = 1
        return bytes([self._observe_tok_counter])

    def _send_dgram(self, datagram):
        """Send a CoAP datagram. Holds the send lock for the
        BIO-drain so two writers can't interleave records."""
        with self._send_lock:
            if self.conn is None:
                raise SessionClosedError()
            send_failed = False
            try:
                self.conn.send(datagram)
                self._last_send_ts = time.monotonic()
                while True:
                    o = self.conn.bio_read(65535)
                    if not o:
                        break
                    for r in _split_dtls(o):
                        if self.sock.send(r) != len(r):
                            raise OSError('incomplete UDP send')
            except SSL.WantReadError:
                pass
            except OSError:
                send_failed = True
            if send_failed:
                raise EndpointError() from OSError('UDP send failed')

    def _reader_loop(self):
        """Pump UDP socket → DTLS BIO → CoAP parser. Demuxes to pending
        / observe handlers. Exits on socket error or stop event."""
        sock = self.sock
        conn = self.conn
        sock.settimeout(self.READER_RECV_TIMEOUT_S)
        try:
            while not self._stop.is_set():
                try:
                    d = sock.recv(65535)
                except socket.timeout:
                    continue
                except OSError as e:
                    if self._stop.is_set():
                        return          # close() got here first
                    if e.errno in _ADVISORY_ERRNOS:
                        logger.debug("reader: advisory %s from %s, continuing",
                                     errno.errorcode.get(e.errno, e.errno),
                                     self.host)
                        continue
                    logger.warning("reader exiting: socket error %s from %s",
                                   errno.errorcode.get(e.errno, e.errno),
                                   self.host)
                    return
                except ValueError:
                    # recv on a socket closed underneath the reader.
                    if not self._stop.is_set():
                        logger.warning("reader exiting: socket closed "
                                       "underneath it")
                    return
                if not d:
                    continue
                # pyOpenSSL's SSL.Connection is not thread-safe — the
                # same SSL object must not be touched by multiple
                # threads concurrently. Drain decrypted records into a
                # local list under _send_lock so the reader never races
                # a sender's conn.send()/bio_read(). Dispatch happens
                # AFTER releasing the lock because _dispatch_coap may
                # call _send_dgram (auto-ACK for CON frames), which
                # re-acquires the lock — holding it across dispatch
                # would deadlock.
                packets = []
                exit_reader = False
                with self._send_lock:
                    try:
                        conn.bio_write(d)
                    except SSL.Error as e:
                        logger.warning("DTLS bio_write: %s", e)
                        return
                    while True:
                        try:
                            pl = conn.recv(65535)
                        except SSL.WantReadError:
                            break
                        except SSL.ZeroReturnError:
                            logger.info("DTLS peer closed connection")
                            exit_reader = True
                            break
                        except SSL.Error as e:
                            logger.warning("DTLS recv: %s", e)
                            exit_reader = True
                            break
                        if not pl:
                            break
                        packets.append(pl)
                for pl in packets:
                    try:
                        self._dispatch_coap(pl)
                    except Exception as e:
                        logger.warning("dispatch: %s", e)
                if exit_reader:
                    return
        finally:
            # Reader no longer owns the socket — callers must fail fast.
            self._reader_running.clear()
            # Make sure pending waiters don't hang if the reader dies.
            for tok, (ev, container) in list(self._pending.items()):
                container.setdefault('err', SessionClosedError())
                ev.set()

    def _dispatch_coap(self, datagram):
        try:
            mt, code, mid, tok, ropts, payload = parse_coap(datagram)
        except Exception as e:
            logger.debug("malformed CoAP: %s", e)
            return

        if DEBUG_BRIDGE:
            kind = ['CON', 'NON', 'ACK', 'RST'][mt]
            logger.info("rx %s code=%s mid=%04x tok=%s opts=%d pl=%d",
                        kind, fmt_code(code), mid, tok.hex() or '-',
                        len(ropts), len(payload))

        # ACK back any CON from the device to suppress retransmits.
        # RFC 7252 §4.2 — ACK is a bare frame (token len 0, code 0).
        if mt == TYPE_CON:
            try:
                self._send_dgram(build_coap(TYPE_ACK, 0, mid, b'', []))
            except Exception as e:
                logger.warning("ACK send: %s", e)

        # Empty ACK with no options & no payload = "separate response
        # coming" — used by Samsung's RT-OCF for the larger reads. Stop
        # the retransmit timer on the client side and wait for the CON.
        if mt == TYPE_ACK and code == 0 and not payload and not ropts:
            return

        # Pending one-shot? Resolve and return.
        rec = self._pending.get(tok)
        if rec is not None:
            ev, container = rec
            container['code']    = code
            container['mtype']   = mt
            container['mid']     = mid
            container['options'] = ropts
            container['payload'] = payload
            ev.set()
            return

        # OBSERVE notification?
        href = self._observe_tokens.get(tok)
        if href is not None:
            if code != 0x45:
                logger.warning("observe %s: non-2.05 %s",
                               href, fmt_code(code))
                return
            cb = self.on_notification
            if cb is not None:
                try:
                    cb(href, payload)
                except Exception as e:
                    logger.warning("notification callback %s: %s",
                                   href, e)
            return

        # Stale token (post-reconnect or unknown) — drop quietly.

    # ---- request primitives ------------------------------------------

    def get(self, path_segs, query=(), timeout=10.0):
        """Token-stable Block2 GET. Returns (code, payload_bytes).

        Reuses one CoAP token across every block of a multi-block
        response — Samsung's server keys per-transfer state on the
        token, and dropping a fresh token on block 1+ silently drops
        the request."""
        self._check_live()
        tok = self._next_tok()
        blob = b''
        num = 0
        last_code = None
        last_opts = []
        deadline = time.time() + timeout
        szx = BLOCK_SZX   # server may negotiate down; track per-transfer
        while True:
            if num > 0:
                self.pace()
            container = {}
            for attempt in range(_BLOCK_MAX_ATTEMPTS):
                ev = threading.Event()
                container = {}
                self._pending[tok] = (ev, container)
                try:
                    mid = self._next_mid()
                    opts = [(URI_PATH, s.encode()) for s in path_segs]
                    for q in query:
                        opts.append((URI_QUERY, q.encode()))
                    opts.append((ACCEPT, CF_CBOR))
                    if num > 0:
                        opts.append((BLOCK2, block_value(num, 0, szx)))
                    self._send_dgram(
                        build_coap(TYPE_CON, METHOD_GET, mid, tok, opts))
                    per_wait = min(_BLOCK_ACK_TIMEOUT,
                                   max(0.1, deadline - time.time()))
                    if ev.wait(per_wait):
                        break  # got a response
                    remaining = deadline - time.time()
                    if remaining <= 0 or attempt == _BLOCK_MAX_ATTEMPTS - 1:
                        logger.debug(
                            "GET %s /%s block %d: timed out after %d attempt(s)",
                            self.host, '/'.join(path_segs), num, attempt + 1,
                        )
                        raise SessionTimeoutError()
                    logger.debug(
                        "GET %s /%s block %d: attempt %d/%d timeout, retrying",
                        self.host, '/'.join(path_segs), num,
                        attempt + 1, _BLOCK_MAX_ATTEMPTS,
                    )
                finally:
                    self._pending.pop(tok, None)
            if 'err' in container:
                raise container['err']

            code = container['code']
            payload = container['payload']
            ropts = container['options']
            last_code = code
            last_opts = ropts
            blob += payload
            # 4.xx / 5.xx responses don't carry Block2 continuation —
            # bail with whatever we got. Caller decides if 4.xx is fatal.
            if code >> 5 != 2:
                return code, blob
            b2 = [v for n, v in ropts if n == BLOCK2]
            more = 0
            if b2:
                bv = int.from_bytes(b2[0], 'big')
                more = (bv >> 3) & 1
                server_szx = bv & 0x07
                if server_szx != szx:
                    szx = server_szx
            if not more:
                break
            num += 1
            if num > self.MAX_BLOCKS:
                raise BlockwiseError()
        return last_code, blob

    def post(self, path_segs, body_cbor, timeout=8.0):
        """Single-frame POST with a CBOR-encoded body. Returns
        (code, payload_bytes). body_cbor must already be encoded."""
        self._check_live()
        tok = self._next_tok()
        mid = self._next_mid()
        opts = [(URI_PATH, s.encode()) for s in path_segs]
        opts.append((CONTENT_FORMAT, CF_CBOR))
        opts.append((ACCEPT, CF_CBOR))
        datagram = build_coap(TYPE_CON, METHOD_POST, mid, tok, opts,
                              body_cbor)
        ev = threading.Event()
        container = {}
        self._pending[tok] = (ev, container)
        try:
            self._send_dgram(datagram)
            if not ev.wait(timeout):
                raise SessionTimeoutError()
            if 'err' in container:
                raise container['err']
            return container['code'], container['payload']
        finally:
            self._pending.pop(tok, None)

    def ping(self):
        """RFC 7252 §4.4 CoAP Ping — empty CON, no token, no payload.
        Fire-and-forget: we do not wait for the matching RST because
        Samsung's RT-OCF doesn't reliably emit one (verified
        2026-06-04: every sync ping timed out while polls succeeded
        at 200+/window). The send itself is the keepalive — it
        tickles Samsung's observer state so OBSERVE subscriptions
        aren't aged out.

        Real half-open-session detection lives in PollScheduler's
        `last_success_ts`, surfaced through KeepaliveTask's
        `liveness_fn`."""
        self._check_live()
        mid = self._next_mid()
        self._send_dgram(build_coap(TYPE_CON, 0, mid, b'', []))
        return mid

    def refresh_observes(self, paths):
        """Drop all current OBSERVE registrations and re-subscribe to
        the given paths. Used as a periodic safety net — CoAP OBSERVE
        has no built-in TTL but Samsung's RT-OCF can age out its
        observer table during cloud blips even while the DTLS session
        stays healthy. Without this, internet recovery on a still-
        reachable device leaves push permanently dead.

        Best-effort: dereg failures are logged and we still bind fresh
        tokens via subscribe. Brief race window where a notify on the
        old token gets dropped as 'stale' — acceptable for a 6h-scale
        safety net."""
        self._check_live()
        for tok, href in list(self._observe_tokens.items()):
            segs = [s for s in href.split('/') if s]
            try:
                self._send_observe_dereg(tok, segs)
            except Exception as e:
                logger.warning("refresh dereg %s: %s", href, e)
        self._observe_tokens.clear()
        time.sleep(0.1)
        for path in paths:
            try:
                self.subscribe(list(path))
                time.sleep(0.05)
            except Exception as e:
                logger.warning("refresh subscribe %s: %s", path, e)

    def subscribe(self, path_segs):
        """Register an OBSERVE on the given path. The initial 2.05
        notification and all subsequent state-change notifications
        will fire on_notification(href, payload_bytes).

        Returns the token used (in case the caller wants to deregister
        later)."""
        self._check_live()
        tok = self._next_observe_tok()
        href = '/' + '/'.join(path_segs)
        # Register the token BEFORE sending — otherwise the device
        # could respond between send() and the dict insert, and the
        # reader thread would drop the initial 2.05 as "stale".
        self._observe_tokens[tok] = href
        mid = self._next_mid()
        opts = [(URI_PATH, s.encode()) for s in path_segs]
        opts.append((OBSERVE, OBSERVE_REGISTER))
        opts.append((ACCEPT, CF_CBOR))
        self._send_dgram(
            build_coap(TYPE_CON, METHOD_GET, mid, tok, opts))
        return tok
