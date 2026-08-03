"""Deterministic baseline checks for the current OCF worker stop contract."""

from __future__ import annotations

import threading

from smartthings_local.ocf.keepalive import KeepaliveTask
from smartthings_local.ocf.observe_refresh import ObserveRefreshTask
from smartthings_local.ocf.poll_scheduler import PollScheduler, PollTier
from smartthings_local.ocf.state_cache import StateCache

_THREAD_DEADLINE_S = 2.0


class _Session:
    def ping(self):
        return None

    def refresh_observes(self, paths):
        return None


class _Descriptor:
    def on_observation(self, state, href, rep):
        return None


class _ObservedEvent(threading.Event):
    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, timeout=None):
        self.waiting.set()
        return super().wait(timeout)


def _assert_worker_stops(target, name: str):
    stop = _ObservedEvent()
    errors: list[str] = []

    def run():
        try:
            target(stop)
        except Exception as error:  # noqa: BLE001  # pragma: no cover
            errors.append(type(error).__name__)

    worker = threading.Thread(target=run, name=name, daemon=True)
    worker.start()
    assert stop.waiting.wait(_THREAD_DEADLINE_S), (
        f"{name} did not enter an interruptible wait"
    )
    stop.set()
    worker.join(_THREAD_DEADLINE_S)
    assert not worker.is_alive(), f"{name} did not stop"
    assert errors == [], f"{name} raised {errors[0]}"


def test_keepalive_worker_stops_without_waiting_for_interval():
    task = KeepaliveTask(_Session(), interval_s=3600.0)
    _assert_worker_stops(task.run_forever, "test-keepalive")


def test_observe_refresh_worker_stops_without_waiting_for_interval():
    task = ObserveRefreshTask(_Session(), [], interval_s=3600.0)
    _assert_worker_stops(task.run_forever, "test-observe-refresh")


def test_poll_scheduler_worker_stops_without_leaking_thread():
    scheduler = PollScheduler(
        _Session(),
        StateCache(_Descriptor()),
        [PollTier("idle", interval_s=3600.0, paths=())],
    )
    _assert_worker_stops(scheduler.run_forever, "test-poll-scheduler")
