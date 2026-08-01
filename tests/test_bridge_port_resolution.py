"""Port-resolution logic for the MQTT bridge: the stateless pre-flight
gate and OCF-band autodiscovery in PushBridge. The DTLS probe is faked so
these run without hardware — only the routing/gating/caching is exercised.
"""
import logging
import time
import types

import pytest

import mqtt_demo.bridge as bridge


def _mk_bridge(ocf_port, default=49155, discovered=None):
    """A PushBridge shell with only the attributes _resolve_port touches,
    bypassing the heavyweight __init__ (MQTT client, cert paths, …)."""
    b = bridge.PushBridge.__new__(bridge.PushBridge)
    b.app = types.SimpleNamespace(ip='10.0.0.9', ocf_port=ocf_port, index=0)
    b.descriptor = types.SimpleNamespace(default_observe_port=default)
    b._discovered_port = discovered
    b.log = logging.getLogger('test-bridge')
    return b


def _fake_probe(live_ports):
    """Return a probe() stand-in reporting is_dtls_server for live_ports."""
    def fake(ip, port, **kw):
        alive = port in live_ports
        return types.SimpleNamespace(
            port=port, is_dtls_server=alive,
            outcome='live' if alive else 'dead')
    return fake


def test_pinned_live_port_is_gated_and_returned(monkeypatch):
    monkeypatch.setattr(bridge, 'probe', _fake_probe({49155}))
    b = _mk_bridge(ocf_port=49155)
    assert b._resolve_port() == 49155


def test_pinned_dead_port_raises_for_backoff(monkeypatch):
    monkeypatch.setattr(bridge, 'probe', _fake_probe(set()))
    b = _mk_bridge(ocf_port=49155)
    with pytest.raises(ConnectionError):
        b._resolve_port()


def test_autodiscovery_finds_and_caches_live_port(monkeypatch):
    # Only 49154 answers; it isn't the descriptor default, so discovery is
    # what finds it — and it must be cached for the next reconnect.
    monkeypatch.setattr(bridge, 'probe', _fake_probe({49154}))
    b = _mk_bridge(ocf_port=None, default=49155)
    assert b._resolve_port() == 49154
    assert b._discovered_port == 49154


def test_autodiscovery_returns_a_live_port(monkeypatch):
    # Early-exit: the first candidate to answer LIVE wins. Real devices
    # expose exactly one DTLS port; if several answer, any live one is a
    # correct result.
    monkeypatch.setattr(bridge, 'probe', _fake_probe({49153, 49155}))
    b = _mk_bridge(ocf_port=None, default=49155)
    assert b._resolve_port() in {49153, 49155}


def test_autodiscovery_early_exits_before_dead_ports_finish(monkeypatch):
    # The live port answers immediately; the dead ports "hang" on their
    # retry budget. Discovery must return at the live port's speed, not
    # block on the slow dead probes.
    def slow_probe(ip, port, **kw):
        if port == 49154:
            return types.SimpleNamespace(
                port=port, is_dtls_server=True, outcome='live')
        time.sleep(0.5)  # a dead port burning its retry budget
        return types.SimpleNamespace(
            port=port, is_dtls_server=False, outcome='dead')
    monkeypatch.setattr(bridge, 'probe', slow_probe)
    b = _mk_bridge(ocf_port=None, default=49155)
    t0 = time.time()
    assert b._resolve_port() == 49154
    assert time.time() - t0 < 0.25   # did not wait out the 0.5s dead probes


def test_cached_live_port_is_reused_without_rediscovery(monkeypatch):
    # Cached 49156 and the default 49155 are both live; the cache-first
    # path must return the cached port, not re-race the band (which would
    # tie-break to the default).
    monkeypatch.setattr(bridge, 'probe', _fake_probe({49155, 49156}))
    b = _mk_bridge(ocf_port=None, default=49155, discovered=49156)
    assert b._resolve_port() == 49156


def test_autodiscovery_all_dead_raises_and_clears_cache(monkeypatch):
    monkeypatch.setattr(bridge, 'probe', _fake_probe(set()))
    b = _mk_bridge(ocf_port=None, discovered=49154)
    with pytest.raises(ConnectionError):
        b._resolve_port()
    assert b._discovered_port is None


def test_candidate_ports_cover_band_plus_default(monkeypatch):
    b = _mk_bridge(ocf_port=None, default=49200)
    cands = b._candidate_ports()
    assert set(bridge.OCF_PORT_BAND) <= set(cands)
    assert 49200 in cands
    assert cands == sorted(cands)
