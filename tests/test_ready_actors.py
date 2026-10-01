import numpy as np
import pytest
import hashlib
from collections import Counter
from dougpu.actors import Ring
from dougpu.replay import PackedSamples
from dougpu.config import TrainConfig
from dougpu.ready_actors import ReadyActorPool


def test_ready_bounded_fairness_accounting_and_versions():
    cfg = TrainConfig(engine='reference', workers=4, envs_per_worker=2, infer_batch=4,
                      ready_first=True, sample_credit=True, replay_version_updates=True,
                      worker_timeout=10)
    cfg.validate()
    pool = ReadyActorPool(cfg, [31, 32, 33, 34])
    samples, actual = 0, Counter()
    def remember(batch):
        for i in range(len(batch)):
            raw = b''.join((batch.data[k][i, :batch.data['lengths'][i]] if k == 'tokens'
                            else batch.data[k][i]).tobytes() for k in sorted(batch.data))
            actual[hashlib.sha256(raw).hexdigest()] += 1
    try:
        for _ in range(160):
            batch, wins = pool.collect()
            if len(batch):
                assert np.all(batch.data['version'] == 123)
            samples += len(batch)
            remember(batch)
            selected = pool.select()
            if not selected:
                batch, _ = pool.collect(block=True)
                samples += len(batch)
                remember(batch)
                continue
            choices = [int(np.argmax(pool.requests[i*2+j]['actions'][:, :15].sum(axis=1)))
                       for i in selected for j in range(2)]
            pool.submit_selected(selected, choices, 123)
            assert len(pool._pending) <= cfg.workers
            assert max(pool.steps)-min(pool.steps) <= 2
            assert len(pool.ready)+len(pool._pending) == cfg.workers
        batch, _ = pool.drain()
        samples += len(batch)
        remember(batch)
        assert samples > 0
        assert min(pool.steps) > 5
        assert pool.counts['submitted'] == pool.counts['received']
        assert not pool._pending and len(set(pool.ready)) == cfg.workers
        expected = Counter()
        for seed, steps in zip([31, 32, 33, 34], pool.steps):
            ring = Ring('reference', 2, seed)
            for _ in range(steps):
                req = ring.requests()
                choices = [int(np.argmax(r['actions'][:, :15].sum(axis=1))) for r in req]
                rows, _ = ring.advance(choices, 123)
                batch = PackedSamples.pack(rows)
                for i in range(len(batch)):
                    raw = b''.join((batch.data[k][i, :batch.data['lengths'][i]] if k == 'tokens'
                                    else batch.data[k][i]).tobytes() for k in sorted(batch.data))
                    expected[hashlib.sha256(raw).hexdigest()] += 1
        assert actual == expected
    finally:
        pool.close()
    assert all(not p.is_alive() for p in pool.processes)


def test_ready_options_are_explicit():
    with pytest.raises(ValueError):
        TrainConfig(ready_first=True).validate()
    with pytest.raises(ValueError):
        TrainConfig(actor_learner_overlap=True).validate()
