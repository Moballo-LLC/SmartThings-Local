"""Compatibility baseline for the published API and LocalThings consumer."""

from __future__ import annotations

import inspect

from smartthings_local.ocf.observe_refresh import ObserveRefreshTask
from smartthings_local.ocf.state_cache import StateCache
from smartthings_local.protocol.dtls_session import DtlsCoapSession


def _parameter_names(callable_object) -> list[str]:
    return list(inspect.signature(callable_object).parameters)


def test_dtls_session_constructor_keeps_file_memory_and_local_port_inputs():
    assert _parameter_names(DtlsCoapSession) == [
        "host",
        "port",
        "cert_path",
        "key_path",
        "cert_pem",
        "key_pem",
        "on_notification",
        "mtu",
        "rate_limit_rps",
        "local_port",
    ]


def test_dtls_session_keeps_current_consumer_methods():
    expected = {
        "close",
        "connect",
        "get",
        "join",
        "pace",
        "ping",
        "post",
        "refresh_observes",
        "start_reader",
        "subscribe",
    }
    assert expected <= set(dir(DtlsCoapSession))
    assert _parameter_names(DtlsCoapSession.get) == [
        "self",
        "path_segs",
        "query",
        "timeout",
    ]
    assert _parameter_names(DtlsCoapSession.post) == [
        "self",
        "path_segs",
        "body_cbor",
        "timeout",
    ]
    assert _parameter_names(DtlsCoapSession.subscribe) == ["self", "path_segs"]


def test_state_cache_keeps_current_consumer_surface():
    assert _parameter_names(StateCache) == ["descriptor"]
    expected = {
        "apply_optimistic",
        "apply_rep",
        "freshness_s",
        "get",
        "index_device_tree",
        "set_on_change",
        "snapshot",
        "stalest",
    }
    assert expected <= set(dir(StateCache))


def test_observe_refresh_task_keeps_current_consumer_surface():
    assert _parameter_names(ObserveRefreshTask) == [
        "session",
        "paths",
        "interval_s",
        "logger",
    ]
    assert _parameter_names(ObserveRefreshTask.run_forever) == ["self", "stop"]
