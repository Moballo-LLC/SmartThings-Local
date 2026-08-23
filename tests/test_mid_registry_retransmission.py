"""Follow-ups to the shared MID registry (#57).

Covers the read path reusing one MID across retransmissions, and the two
identifier-allocation failures now carrying their own type.
"""

from __future__ import annotations

import threading

import pytest

from smartthings_local.errors import (
    SessionIdentifierError,
    SessionTimeoutError,
)
from smartthings_local.protocol import dtls_session
from smartthings_local.protocol.coap import (
    METHOD_GET,
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


def test_get_retransmission_reuses_one_mid_and_datagram(monkeypatch):
    """RFC 7252 §4.2: a retransmission is the same message.

    A fresh MID per attempt presents each retry to the appliance as a new
    request it may answer separately.
    """
    session = _session()
    sent = []

    monkeypatch.setattr(dtls_session, "_BLOCK_ACK_TIMEOUT", 0.01)

    def send(datagram):
        # Skip the bare ACK the session sends back for the device's own CON
        # notification; only the GET requests are under test here.
        if parse_coap(datagram)[1] == METHOD_GET:
            sent.append(datagram)

    session._send_dgram = send

    # Answer only once the third attempt is on the wire, so two full
    # retransmissions have to happen first.
    def wait_live(_event, _timeout):
        if len(sent) < 3:
            return False
        _mtype, _code, mid, token, _options, _payload = parse_coap(sent[-1])
        session._dispatch_coap(
            build_coap(TYPE_CON, 0x45, mid + 1, token, [], b"ok")
        )
        return True

    session._wait_live = wait_live

    assert session.get(["device", "0"], timeout=5.0) == (0x45, b"ok")

    assert len(sent) == 3
    assert len(set(sent)) == 1, "retransmissions must be byte-identical"
    mids = {parse_coap(datagram)[2] for datagram in sent}
    assert len(mids) == 1
    assert session._pending == {}
    assert session._pending_mids == {}


def test_exhausted_attempts_still_release_the_single_registration():
    session = _session()
    session._send_dgram = lambda _datagram: None
    session._wait_live = lambda _event, _timeout: False

    with pytest.raises(SessionTimeoutError):
        session.get(["device", "0"], timeout=0.05)

    assert session._pending == {}
    assert session._pending_mids == {}


def test_token_already_in_flight_is_its_own_error():
    session = _session()
    event = threading.Event()
    container = {}
    token = b"\x01\x02"
    session._register_pending_request(token, event, container)

    with pytest.raises(SessionIdentifierError):
        session._register_pending_request(token, threading.Event(), {})


def test_mid_allocation_probes_only_past_the_live_exchanges():
    """Pigeonhole: one more candidate than live exchanges always suffices.

    The old walk went through all 65,536 identifiers to discover what the
    registry size already implied.
    """
    session = _session()
    probes = []
    real_pending_mids = session._pending_mids

    class _CountingMids(dict):
        def __contains__(self, key):
            probes.append(key)
            return dict.__contains__(self, key)

    counting = _CountingMids(real_pending_mids)
    for offset in range(1, 4):
        counting[(session._mid + offset) & 0xFFFF] = object()
    session._pending_mids = counting

    mid = session._next_mid()

    assert mid == (session._mid) & 0xFFFF
    assert len(probes) == 4, "three live exchanges, so at most four probes"
    assert mid not in counting


def test_mid_allocation_raises_when_every_identifier_is_live():
    session = _session()
    session._pending_mids = {mid: object() for mid in range(0x10000)}

    with pytest.raises(SessionIdentifierError):
        session._next_mid()
