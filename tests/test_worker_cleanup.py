"""Deterministic baseline checks for the current OCF worker stop contract."""

from __future__ import annotations

import threading

from smartthings_local.ocf.keepalive import KeepaliveTask
from smartthings_local.ocf.observe_refresh import ObserveRefreshTask
from smartthings_local.ocf.poll_scheduler import PollScheduler, PollTier
from smartthings_local.ocf.state_cache import StateCache


class _Session:
    def ping(self):
        return None

    def refresh_observes(self, paths):
        return None


class _Descriptor:
    def on_observation(self, state, href, rep):
        return None


def _assert_worker_stops(target, name: str):
    stop = threading.Event()
    entered = threading.Event()
    errors: list[str] = []

    def run():
        entered.set()
        try:
            target(stop)
        except Exception as error:  # noqa: BLE001  # pragma: no cover
            errors.append(type(error).__name__)

    worker = threading.Thread(target=run, name=name)
    worker.start()
    assert entered.wait(0.5), f"{name} did not enter its worker"
    stop.set()
    worker.join(0.5)
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
