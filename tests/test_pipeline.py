from copy import deepcopy
from dataclasses import replace
import time

import numpy as np
import pytest

from dougpu.actors import ActorPool, EvaluationPool, Ring
from dougpu.config import TrainConfig
from dougpu.replay import Replay, ReplayPrefetch


def assert_request(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize('groups', [1, 2, 4])
def test_grouped_round_preserves_samples_requests_and_rng(groups):
    cfg = TrainConfig(engine='reference', workers=4, envs_per_worker=2)
    seeds = [32, 33, 34, 35]
    pool = ActorPool(cfg, seeds)
    rings = [Ring('reference', 2, seed) for seed in seeds]
    direct_rng, grouped_rng = np.random.default_rng(59), np.random.default_rng(59)
    expected, actual = Replay(419), Replay(419)

    def choose(requests, rng):
        return [int(rng.integers(len(r['actions']))) if rng.random() < .3 else
                int(np.argmax(r['actions'][:, :15].sum(axis=1))) for r in requests]

    try:
        for step in range(120):
            requests = [r for ring in rings for r in ring.requests()]
            for a, b in zip(requests, pool.requests):
                assert_request(a, b)
            choices = choose(requests, direct_rng)
            rows, wins = [], []
            for i, ring in enumerate(rings):
                batch, winners = ring.advance(choices[i*2:i*2+2], step//10)
                rows.extend(batch)
                wins.extend(winners)
            for group in range(groups):
                first, last = group*4//groups, (group+1)*4//groups
                selected = choose(pool.requests[first*2:last*2], grouped_rng)
                pool.submit(selected, step//10, first, last)
            packed, got_wins = pool.receive()
            assert wins == got_wins and len(rows) == len(packed)
            assert direct_rng.bit_generator.state == grouped_rng.bit_generator.state
            expected.add(rows)
            actual.add(packed)
        for key in expected.data:
            np.testing.assert_array_equal(expected.data[key], actual.data[key])
    finally:
        pool.close()
    assert all(not p.is_alive() for p in pool.processes)


def test_actor_protocol_rejects_partial_duplicate_and_busy_evaluation():
    pool = ActorPool(TrainConfig(engine='reference', workers=2, envs_per_worker=1), [71, 72])
    try:
        for choices, first, last in [([0], 0, 2), ([0], -1, 0), ([], 1, 1), ([0], 2, 3)]:
            with pytest.raises(ValueError):
                pool.submit(choices, 0, first, last)
            assert not pool._pending
        pool.submit([0], 0, 0, 1)
        with pytest.raises(RuntimeError):
            pool.submit([0], 0, 0, 1)
        with pytest.raises(RuntimeError):
            pool.receive()
        with pytest.raises(RuntimeError):
            EvaluationPool(pool, [(0, 1)], 1, False)
        pool.submit([0], 0, 1, 2)
        pool.receive()
        assert not pool._pending
    finally:
        pool.close()


def test_ready_first_receive_still_merges_in_worker_order(monkeypatch):
    import dougpu.actors as actors
    original = actors.wait
    orders = []

    def reversed_ready(conns, timeout):
        ready = original(conns, timeout)
        ready = sorted(ready, key=lambda c: pool.conns.index(c), reverse=True)
        orders.extend(pool.conns.index(c) for c in ready)
        return ready

    pool = ActorPool(TrainConfig(engine='reference', workers=2, envs_per_worker=1), [15, 16])
    rings = [Ring('reference', 1, seed) for seed in (15, 16)]
    try:
        pool.submit([0, 0], 0)
        assert all(conn.poll(5) for conn in pool.conns)
        monkeypatch.setattr(actors, 'wait', reversed_ready)
        pool.receive()
        assert orders == [1, 0]
        for ring, actual in zip(rings, pool.requests):
            ring.requests()
            ring.advance([0], 0)
            assert_request(ring.requests()[0], actual)
    finally:
        pool.close()


def test_actor_round_deadline_does_not_scale_with_worker_count(monkeypatch):
    import dougpu.actors as actors
    pool = ActorPool(TrainConfig(engine='reference', workers=2, envs_per_worker=1), [9, 10])
    try:
        pool.submit([0, 0], 0)
        pool._deadline = time.monotonic() - 1
        monkeypatch.setattr(actors, 'wait', lambda conns, timeout: [])
        with pytest.raises(TimeoutError):
            pool.receive()
    finally:
        pool.close()


def populated_replay():
    replay = Replay(700)
    rows = [(np.arange(1, i % 27 + 2, dtype=np.uint8), np.full(21, i / 7),
             np.full(16, i / 9), i % 3, np.full(30, i / 11), float(i % 2), i//100)
            for i in range(650)]
    replay.add(rows)
    return replay


@pytest.mark.parametrize('reuse_strata', [False, True])
def test_prefetch_matches_direct_batches_and_consumed_rng_only(reuse_strata):
    replay, cfg = populated_replay(), TrainConfig(micro_batch=8, accumulation=2)
    direct, main = np.random.default_rng(91), np.random.default_rng(91)
    bins = replay.strata(cfg, 8) if reuse_strata else None
    ahead = ReplayPrefetch(replay, cfg, main, 8, lambda batch: batch, bins)
    for _ in range(7):
        before = deepcopy(main.bit_generator.state)
        ahead.prepare()
        assert before == main.bit_generator.state
        with pytest.raises(RuntimeError):
            ahead.prepare()
        actual, length = ahead.take()
        expected = replay.sample(cfg, direct, 8)
        assert length == expected['tokens'].shape[-1]
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        assert main.bit_generator.state == direct.bit_generator.state
        with pytest.raises(RuntimeError):
            ahead.take()
    before = deepcopy(main.bit_generator.state)
    ahead.prepare()
    del ahead  # stop/checkpoint: unused speculation must not advance the main RNG
    assert main.bit_generator.state == before


@pytest.mark.parametrize('version_updates', [False, True])
@pytest.mark.parametrize('wrapped', [False, True])
def test_phase_strata_preserve_order_samples_and_rng(monkeypatch, version_updates, wrapped):
    replay = populated_replay()
    if wrapped:
        replay.add([(np.array([1, 2], np.uint8), np.zeros(21), np.zeros(16), i % 3,
                     np.zeros(30), float(i % 2), 8) for i in range(400)])
    cfg = TrainConfig(micro_batch=8, accumulation=2, replay_max_age=1,
                      replay_version_updates=version_updates)
    version = 6
    age = cfg.replay_max_age * (cfg.updates_per_cycle if version_updates else 1)
    ids = np.arange(replay.size)
    ids = ids[replay.data['version'][:replay.size] >= version-age]
    bins = replay.strata(cfg, version)
    for role, bucket in enumerate(bins):
        np.testing.assert_array_equal(bucket, ids[replay.data['role'][ids] == role])
    direct, reused = np.random.default_rng(91), np.random.default_rng(91)
    expected = [replay.sample(cfg, direct, version) for _ in range(4)]
    monkeypatch.setattr(replay, 'eligible', lambda *_: pytest.fail('Repeated replay scan'))
    for want in expected:
        got = replay.sample(cfg, reused, version, bins)
        for key in want:
            np.testing.assert_array_equal(got[key], want[key])
    assert direct.bit_generator.state == reused.bit_generator.state


def test_strata_rebuilt_after_versions_and_replay_change():
    replay = populated_replay()
    cfg, rng = TrainConfig(replay_max_age=0), np.random.default_rng(9)
    assert all(len(bucket) for bucket in replay.strata(cfg, 6))
    assert not any(len(bucket) for bucket in replay.strata(cfg, 7))
    with pytest.raises(RuntimeError, match='No fresh enough'):
        replay.sample(cfg, rng, 7)
    rows = [(np.array([1], np.uint8), np.zeros(21), np.zeros(16), role,
             np.zeros(30), 1., 7) for role in range(3)]
    replay.add(rows[:1])
    with pytest.raises(RuntimeError, match='All three role strata'):
        replay.sample(cfg, rng, 7, replay.strata(cfg, 7))
    replay.add(rows[1:])
    bins = replay.strata(cfg, 7)
    assert [len(bucket) for bucket in bins] == [1, 1, 1]
    replay.sample(cfg, rng, 7, bins)
    restored = Replay(replay.capacity)
    restored.restore(replay.export())
    for a, b in zip(bins, restored.strata(cfg, 7)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('failure', ['sample', 'transfer'])
def test_prefetch_defers_failure_without_committing_unused_rng(monkeypatch, failure):
    replay, cfg, main = populated_replay(), TrainConfig(), np.random.default_rng(59)
    calls = []

    def transfer(batch):
        calls.append(1)
        if failure == 'transfer' and len(calls) == 2:
            raise ValueError('Injected transfer failure')
        return batch

    ahead = ReplayPrefetch(replay, cfg, main, 8, transfer)
    ahead.prepare()
    ahead.take()
    consumed = deepcopy(main.bit_generator.state)
    if failure == 'sample':
        def fail(*args):
            raise ValueError('Injected sampling failure')
        monkeypatch.setattr(replay, 'sample', fail)
    ahead.prepare()  # must not interrupt an already dispatched optimizer step
    assert main.bit_generator.state == consumed
    with pytest.raises(ValueError, match='Injected'):
        ahead.take()
    assert main.bit_generator.state == consumed


def test_group_config_validation():
    cfg = TrainConfig(workers=4)
    for groups in (0, -1, 3, 8):
        with pytest.raises(ValueError):
            replace(cfg, actor_groups=groups).validate()
    for groups in (1, 2, 4):
        replace(cfg, actor_groups=groups).validate()
