import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from dougpu.config import ModelConfig, TrainConfig
from dougpu.replay import PackedSamples
from dougpu.checkpoint import Store


@pytest.mark.parametrize('ready,credit,overlap', [(False, False, False), (False, True, False),
                                               (True, True, False), (True, True, True)])
def test_training_accounts_ordered_blocking_collect_and_drain(tmp_path, monkeypatch, ready, credit, overlap):
    from dougpu import train, inference, ready_actors

    def samples(count):
        return PackedSamples.pack([(np.ones(6, np.uint8), np.zeros(21), np.zeros(16),
                                    i % 3, np.zeros(30), 1., 0) for i in range(count)])

    class Pool:
        requests = [None] * 3
        counts = {}
        steps = [0]

        def __init__(self, *args, **kwargs):
            self.collected = 0

        def submit(self, *args):
            pass

        def receive(self):
            return samples(6), [0, 1]

        def collect(self, block=False):
            self.collected += 1
            return (samples(0), []) if self.collected == 1 else (samples(3), [0])

        def select(self):
            return []

        def drain(self):
            return samples(3), [1]

        def close(self):
            pass

    monkeypatch.setattr(train, 'ActorPool', Pool)
    monkeypatch.setattr(ready_actors, 'ReadyActorPool', Pool)
    monkeypatch.setattr(inference, 'Inference', lambda *a, **kw:
                        SimpleNamespace(choose=lambda *a, **kw: [0] * 3, last_stats={}))
    mc = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16, bf16=False)
    tc = TrainConfig(engine='reference', backend='cuda' if jax.default_backend() == 'gpu' else 'cpu',
                     require_tpu=False, workers=1, envs_per_worker=3, micro_batch=3, accumulation=1,
                     replay_capacity=16, fresh_samples=6, updates_per_cycle=1, max_cycles=1,
                     max_hours=1, eval_every=0, ready_first=ready, sample_credit=credit,
                     replay_version_updates=ready, actor_learner_overlap=overlap)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'model': asdict(mc), 'train': asdict(tc)}))
    work = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(config), '--workdir', str(work)])
    train.main()
    saved = Store(work / 'checkpoints').load_latest()
    meta = saved['meta']
    expected = 9 if ready else 6
    assert (meta['cycle'], meta['updates'], meta['frames'], meta['games'], meta['complete_samples']) == (
        1, 1, 0 if ready else 3, 3 if ready else 2, expected)
    assert meta['collection_credit'] == (3 if ready else 0)
    events = [json.loads(line) for line in saved['log'].splitlines()]
    event = next(event for event in events if event['event'] == 'train')
    assert event['fresh_samples'] == event['replay_size'] == expected
    if ready:
        assert event['replay_write_seconds'] == 0
