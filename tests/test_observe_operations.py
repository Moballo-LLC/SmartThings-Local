"""Targeted Observe refresh, unsubscribe, and cleanup contracts."""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from smartthings_local.errors import EndpointError, SessionClosedError
from smartthings_local.protocol.coap import (
    OBSERVE,
    URI_PATH,
    URI_QUERY,
    parse_coap,
)
from smartthings_local.protocol.dtls_session import DtlsCoapSession


class _NullAuth:
    def configure_context(self, _context):
        return None


class _Socket:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _session():
    session = DtlsCoapSession(
        "device.example",
        5684,
        auth=_NullAuth(),
        rate_limit_rps=1_000_000,
    )
    session.conn = object()
    return session


def _add_relation(session, path, query=()):
    session._send_dgram = Mock()
    return session.subscribe(path, query=query)


def _options(datagram, number):
    return tuple(value for option, value in parse_coap(datagram)[4]
                 if option == number)


def test_refresh_targets_only_requested_path_and_preserves_sibling():
    session = _session()
    mode = ["mode", "vs", "0"]
    door = ["door", "vs", "0"]
    mode_token = _add_relation(session, mode, ("if=oic.if.a",))
    door_token = _add_relation(session, door, ("if=oic.if.s",))
    requests = []
    session._send_dgram = requests.append

    successful, failures = session.refresh_observes((mode,))

    assert successful == ("/mode/vs/0",)
    assert failures == 0
    assert mode_token not in session._observe_tokens
    assert session._observe_tokens[door_token] == "/door/vs/0"
    assert session._observe_queries[door_token] == ("if=oic.if.s",)
    assert len(requests) == 2
    assert _options(requests[0], URI_PATH) == (b"mode", b"vs", b"0")
    assert _options(requests[0], URI_QUERY) == (b"if=oic.if.a",)
    assert _options(requests[0], OBSERVE) == (b"\x01",)
    assert _options(requests[1], URI_QUERY) == (b"if=oic.if.a",)
    assert _options(requests[1], OBSERVE) == (b"",)


def test_refresh_override_replaces_all_old_query_variants_once():
    session = _session()
    path = ["mode", "vs", "0"]
    _add_relation(session, path, ("if=oic.if.a",))
    _add_relation(session, path, ("if=oic.if.s",))
    requests = []
    session._send_dgram = requests.append

    result = session.refresh_observes(
        (path,),
        queries_by_href={"/mode/vs/0": ("if=oic.if.b",)},
    )

    assert result == (("/mode/vs/0",), 0)
    assert len(requests) == 3
    assert {
        _options(datagram, URI_QUERY) for datagram in requests[:2]
    } == {(b"if=oic.if.a",), (b"if=oic.if.s",)}
    assert _options(requests[2], URI_QUERY) == (b"if=oic.if.b",)


def test_refresh_reports_registration_sends_without_claiming_confirmation():
    session = _session()
    session.subscribe = Mock(
        side_effect=(b"\x41", EndpointError(), b"\x43")
    )

    successful, failures = session.refresh_observes(
        (
            ("door", "cooler", "0"),
            ("door", "freezer", "0"),
            ("mode", "vs", "0"),
        )
    )

    assert successful == ("/door/cooler/0", "/mode/vs/0")
    assert failures == 1
    assert session.subscribe.call_count == 3


@pytest.mark.parametrize(
    ("paths", "queries_by_href", "error"),
    (
        ((("mode", "vs", "0"), ("door", 7)), None, TypeError),
        ((("mode", "vs", "0"),), {"/other": ()}, ValueError),
        ((("mode", "vs", "0"),), {"/mode/vs/0": (b"bad",)}, TypeError),
        ("/mode/vs/0", None, TypeError),
    ),
)
def test_refresh_validation_is_atomic(paths, queries_by_href, error):
    session = _session()
    token = _add_relation(session, ["door", "vs", "0"])
    original_tokens = dict(session._observe_tokens)
    session.pace = Mock()
    session._send_observe_dereg = Mock()

    with pytest.raises(error):
        session.refresh_observes(
            paths,
            queries_by_href=queries_by_href,
        )

    assert session._observe_tokens == original_tokens
    assert token in session._observe_tokens
    session.pace.assert_not_called()
    session._send_observe_dereg.assert_not_called()


def test_refresh_dereg_failure_retires_every_old_relation_and_continues():
    session = _session()
    path = ["mode", "vs", "0"]
    first = _add_relation(session, path, ("if=oic.if.a",))
    second = _add_relation(session, path, ("if=oic.if.s",))
    for token in (first, second):
        session._observe_plain_response_mids[token] = 10
        session._legacy_observe_tokens.add(token)
        session._legacy_observe_mids[token] = 11
        session._observe_sequences[token] = (12, 13.0)
    session._send_observe_dereg = Mock(
        side_effect=(EndpointError(), None)
    )
    session.subscribe = Mock(return_value=b"\x50")

    assert session.refresh_observes((path,)) == (("/mode/vs/0",), 0)

    assert session._send_observe_dereg.call_count == 2
    for token in (first, second):
        assert token not in session._observe_tokens
        assert token not in session._observe_queries
        assert token not in session._observe_plain_response_mids
        assert token not in session._legacy_observe_tokens
        assert token not in session._legacy_observe_mids
        assert token not in session._observe_sequences


def test_refresh_paces_each_deregister_and_registration_exactly_once():
    session = _session()
    path = ["mode", "vs", "0"]
    _add_relation(session, path)
    session.pace = Mock()
    session._send_dgram = Mock()

    session.refresh_observes((path,))

    assert session.pace.call_count == 2
    assert session._send_dgram.call_count == 2


def test_refresh_serializes_a_concurrent_subscription():
    session = _session()
    path = ["mode", "vs", "0"]
    _add_relation(session, path)
    deregister_started = threading.Event()
    release_deregister = threading.Event()
    subscribe_started = threading.Event()
    subscribe_finished = threading.Event()
    outcomes = {}

    def deregister(*_args):
        deregister_started.set()
        assert release_deregister.wait(1.0)

    def refresh():
        outcomes["refresh"] = session.refresh_observes((path,))

    def subscribe():
        subscribe_started.set()
        outcomes["subscribe"] = session.subscribe(("door", "vs", "0"))
        subscribe_finished.set()

    session._send_observe_dereg = deregister
    session._send_dgram = Mock()
    refresh_thread = threading.Thread(target=refresh)
    subscribe_thread = threading.Thread(target=subscribe)
    refresh_thread.start()
    assert deregister_started.wait(1.0)
    subscribe_thread.start()
    assert subscribe_started.wait(1.0)
    assert not subscribe_finished.wait(0.05)

    release_deregister.set()
    refresh_thread.join(1.0)
    subscribe_thread.join(1.0)

    assert not refresh_thread.is_alive()
    assert not subscribe_thread.is_alive()
    assert outcomes["refresh"] == (("/mode/vs/0",), 0)
    assert isinstance(outcomes["subscribe"], bytes)


def test_refresh_deduplicates_requested_paths():
    session = _session()
    session.subscribe = Mock(return_value=b"\x41")
    path = ("mode", "vs", "0")

    result = session.refresh_observes((path, path))

    assert result == (("/mode/vs/0",), 0)
    session.subscribe.assert_called_once_with(path, query=())


def test_unsubscribe_retires_all_exact_path_queries_and_preserves_sibling():
    session = _session()
    target = ["mode", "vs", "0"]
    first = _add_relation(session, target, ("if=oic.if.a",))
    second = _add_relation(session, target, ("if=oic.if.s",))
    sibling = _add_relation(session, ["door", "vs", "0"])
    for token in (first, second):
        session._observe_plain_response_mids[token] = 10
        session._legacy_observe_tokens.add(token)
        session._legacy_observe_mids[token] = 11
        session._observe_sequences[token] = (12, 13.0)
    session._refetch_pending = {
        ("/mode/vs/0", ("if=oic.if.a",), True): 1,
        ("/door/vs/0", (), False): 2,
    }
    session.pace = Mock()
    session._send_observe_dereg = Mock()

    assert session.unsubscribe(target) == 2

    assert session.pace.call_count == 2
    assert session._send_observe_dereg.call_args_list == [
        ((first, tuple(target), ("if=oic.if.a",)), {}),
        ((second, tuple(target), ("if=oic.if.s",)), {}),
    ]
    for token in (first, second):
        assert token not in session._observe_tokens
        assert token not in session._observe_queries
        assert token not in session._observe_plain_response_mids
        assert token not in session._legacy_observe_tokens
        assert token not in session._legacy_observe_mids
        assert token not in session._observe_sequences
    assert session._observe_tokens[sibling] == "/door/vs/0"
    assert tuple(session._refetch_pending) == (
        ("/door/vs/0", (), False),
    )


def test_unsubscribe_retires_every_match_before_reraising_first_error():
    session = _session()
    target = ["mode", "vs", "0"]
    first = _add_relation(session, target, ("if=oic.if.a",))
    second = _add_relation(session, target, ("if=oic.if.s",))
    sibling = _add_relation(session, ["door", "vs", "0"])
    first_error = EndpointError()
    session._send_observe_dereg = Mock(
        side_effect=(first_error, EndpointError())
    )

    with pytest.raises(EndpointError) as raised:
        session.unsubscribe(target)

    assert raised.value is first_error
    assert session._send_observe_dereg.call_count == 2
    assert first not in session._observe_tokens
    assert second not in session._observe_tokens
    assert sibling in session._observe_tokens


def test_unsubscribe_validation_has_no_side_effects():
    session = _session()
    token = _add_relation(session, ["mode", "vs", "0"])
    session.pace = Mock()
    session._send_observe_dereg = Mock()

    with pytest.raises(TypeError):
        session.unsubscribe("/mode/vs/0")

    assert token in session._observe_tokens
    session.pace.assert_not_called()
    session._send_observe_dereg.assert_not_called()


def test_normal_close_paces_every_exact_deregister_and_clears_state(
    monkeypatch,
):
    session = _session()
    session.sock = _Socket()
    first = _add_relation(
        session, ["mode", "vs", "0"], ("if=oic.if.a",)
    )
    second = _add_relation(session, ["door", "vs", "0"])
    session._legacy_observe_tokens.add(first)
    session._observe_sequences[second] = (1, 2.0)
    session.pace = Mock()
    session._send_observe_dereg = Mock()
    monkeypatch.setattr(
        DtlsCoapSession,
        "_send_close_notify",
        Mock(),
    )
    monkeypatch.setattr(
        "smartthings_local.protocol.dtls_session.time.sleep",
        Mock(),
    )

    session.close()

    assert session.pace.call_count == 2
    assert session._send_observe_dereg.call_args_list == [
        ((first, ["mode", "vs", "0"], ("if=oic.if.a",)), {}),
        ((second, ["door", "vs", "0"], ()), {}),
    ]
    assert session._observe_tokens == {}
    assert session._observe_queries == {}
    assert session._legacy_observe_tokens == set()
    assert session._observe_sequences == {}


def test_quiesced_close_paces_every_deregister_for_orderly_teardown():
    session = _session()
    session.sock = _Socket()
    token = _add_relation(session, ["mode", "vs", "0"])
    session.pace = Mock()
    session._pace_orderly_close = Mock()
    session._send_observe_dereg = Mock()

    session.quiesce_for_close()
    session.close()

    session.pace.assert_not_called()
    session._pace_orderly_close.assert_called_once_with()
    session._send_observe_dereg.assert_called_once_with(
        token,
        ["mode", "vs", "0"],
        (),
    )


def test_quiesced_close_preserves_existing_send_override_signature():
    session = _session()
    session.sock = _Socket()
    token = _add_relation(session, ["mode", "vs", "0"])
    sent = []
    session._send_dgram = sent.append
    session._pace_orderly_close = Mock()

    session.quiesce_for_close()
    session.close()

    assert len(sent) == 1
    assert parse_coap(sent[0])[3] == token


def test_orderly_close_send_permission_is_thread_local():
    session = _session()
    session.quiesce_for_close()
    outcomes = []

    def try_application_send():
        try:
            session._send_dgram(b"application request")
        except Exception as error:  # noqa: BLE001 - captured for assertion
            outcomes.append(error)

    def during_teardown_send(_token, _path, _query):
        thread = threading.Thread(target=try_application_send)
        thread.start()
        thread.join()

    session._send_observe_dereg = during_teardown_send
    session._send_observe_dereg_after_quiesce(
        b"o", ["mode", "vs", "0"], ()
    )

    assert len(outcomes) == 1
    assert isinstance(outcomes[0], SessionClosedError)


def test_reader_exit_clears_all_relation_state():
    session = _session()
    token = _add_relation(session, ["mode", "vs", "0"])
    session._observe_plain_response_mids[token] = 10
    session._legacy_observe_tokens.add(token)
    session._legacy_observe_mids[token] = 11
    session._observe_sequences[token] = (12, 13.0)
    session.sock = Mock()
    session._stop.set()

    session._reader_loop()

    assert session._observe_tokens == {}
    assert session._observe_queries == {}
    assert session._observe_plain_response_mids == {}
    assert session._legacy_observe_tokens == set()
    assert session._legacy_observe_mids == {}
    assert session._observe_sequences == {}


def test_running_refetch_drops_result_after_relation_is_retired():
    delivered = []
    session = _session()
    session.on_notification = lambda href, payload: delivered.append(
        (href, payload)
    )
    token = _add_relation(
        session, ["mode", "vs", "0"], ("if=oic.if.a",)
    )
    session._observe_sequences[token] = (1, 2.0)

    def retire_during_read(*_args, **_kwargs):
        with session._state_lock:
            session._retire_observe_token_locked(token)
        return 0x45, b"complete", 2, b"fresh"

    session._blockwise_get = retire_during_read

    session._refetch_one(
        ("/mode/vs/0", ("if=oic.if.a",), False),
        1,
    )

    assert delivered == []
