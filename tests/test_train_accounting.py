import json
from pathlib import Path
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

    received_seeds = []

    class Pool:
        requests = [None] * 3
        counts = {}
        steps = [0]

        def __init__(self, *args, **kwargs):
            received_seeds.append(args[1].tolist())
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
                     replay_capacity=16, fresh_samples=6, updates_per_cycle=1, max_cycles=1, target_updates=1,
                     max_hours=1, eval_every=0, ready_first=ready, sample_credit=credit,
                     replay_version_updates=ready, actor_learner_overlap=overlap)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'model': asdict(mc), 'train': asdict(tc)}))
    work = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(config), '--workdir', str(work)])
    lock_path = tmp_path / 'source_lock.json'
    lock = {'engine': 'reference', 'encoding_schema': 1, 'doutpu_sha256': 'historical'}
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--source-lock', str(lock_path)])
    provenance = {'git_commit': 'unknown', 'git_dirty': 'unknown' if ready else True}
    monkeypatch.setattr(train, 'git_source', lambda _: provenance)
    import run_local
    (work/'source').mkdir(parents=True)
    run_local.archive_trainer(work)
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

    assert {k: meta['versions'][k] for k in provenance} == provenance
    actor = next(event for event in events if event['event'] == 'actor_start')
    assert actor['worker_seeds'] == received_seeds[0]
    expected_rng = np.random.default_rng(tc.seed)
    assert received_seeds[0] == expected_rng.integers(0, 2**32-1, tc.workers, dtype=np.uint64).tolist()
    assert actor['packed'] == (not ready)
    assert actor['worker_order'] == list(range(tc.workers))
    assert actor['mode'] == ('ready_first' if ready else 'ordered')
    assert all(actor[k] == v for k, v in meta['actor_start'].items())
    assert meta['runtime_source']['snapshot_sha256']
    start = next(event for event in events if event['event'] == 'start')
    assert start['versions'] == meta['versions']
    assert start['resume_input'] is meta['resume_input'] is None
    assert start['session_id'] == meta['session_id']
    from scripts.protocol_gate import audit_execution
    plan = {'checkpoint_sha256': saved['sha256'], 'sessions': [dict(
        start_updates=0, end_updates=1, input_checkpoint_sha256=None,
        source_sha256=meta['runtime_source']['snapshot_sha256'],
        model=meta['model'], train=meta['train'], source_lock=lock)]}
    assert audit_execution(work, plan)['status'] == 'PASS'
    # Keep a portable copy and verify the same bytes after a backup restore.
    import shutil
    portable = tmp_path/'evidence'
    shutil.copytree(work, portable, ignore=shutil.ignore_patterns('jax_cache'))
    (portable/'approved-plan.json').write_text(json.dumps(plan))
    assert audit_execution(portable, plan)['status'] == 'PASS'
    plan['sessions'][0]['input_checkpoint_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='resume input'):
        audit_execution(work, plan)
    path = Path(saved['path'])
    before = path.read_bytes()
    monkeypatch.setattr(train, 'git_source', lambda _: {'git_commit': 'new-session', 'git_dirty': True})
    train.main()  # Session provenance can change while the historical lock stays intact.
    restored = Store(work / 'checkpoints').load_latest()
    assert restored['meta']['source_lock'] == lock
    resume = restored['meta']['resume_input']
    assert resume == dict(path=saved['path'], sha256=saved['sha256'],
                          updates=meta['updates'], cycle=meta['cycle'])
    starts = [json.loads(line) for line in restored['log'].splitlines()
              if json.loads(line)['event'] == 'start']
    assert starts[-1]['resume_input'] == resume
    assert starts[-1]['session_id'] == restored['meta']['session_id']
    assert restored['meta']['updates'] == meta['updates']
    assert restored['meta']['versions']['git_commit'] == 'new-session'
    assert restored['meta']['numpy_rng'] == meta['numpy_rng']
    for group in ('params', 'champion', 'replay'):
        for key in saved[group]:
            np.testing.assert_array_equal(saved[group][key], restored[group][key])
    for group in ('m', 'v'):
        for key in saved['optimizer'][group]:
            np.testing.assert_array_equal(saved['optimizer'][group][key], restored['optimizer'][group][key])
    assert path.read_bytes() == before

    # Restarting at the completed boundary creates no actors and consumes no RNG.
    assert len(received_seeds) == 1
    assert restored['meta']['actor_start'] is None

    expected_rng.bit_generator.state = restored['meta']['numpy_rng']
    expected_seeds = expected_rng.integers(0, 2**32-1, tc.workers, dtype=np.uint64).tolist()
    tc.max_cycles = tc.target_updates = 2
    config.write_text(json.dumps({'model': asdict(mc), 'train': asdict(tc)}))
    train.main()
    assert received_seeds[-1] == expected_seeds
    assert len(received_seeds) == 2


def test_source_identity_rejects_tampered_snapshot(tmp_path):
    import zipfile
    from dougpu import train
    from dougpu.runtime import source_identity
    import run_local
    (tmp_path/'source').mkdir()
    run_local.archive_trainer(tmp_path)
    snapshot = tmp_path/'source/trainer_source.zip'
    identity = source_identity(run_local.ROOT, snapshot)
    assert identity['loaded_modules']['dougpu.train'] == 'dougpu/train.py'
    with zipfile.ZipFile(snapshot) as z:
        content = {n: z.read(n) for n in z.namelist()}
    content['dougpu/train.py'] += b'\n# tampered\n'
    with zipfile.ZipFile(snapshot, 'w') as z:
        for name, data in content.items():
            z.writestr(name, data)
    with pytest.raises(ValueError, match='Prepared source mismatch'):
        source_identity(run_local.ROOT, snapshot)
