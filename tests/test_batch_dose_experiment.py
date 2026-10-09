"""Focused gates: no real self-play or long training is performed."""
import json
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from dougpu.checkpoint import Store, sha256_file
from dougpu.config import ModelConfig, TrainConfig
from dougpu.replay import Replay
from dougpu.semantics import expected_parameter_shapes, check_training_semantics
from fork_run import fork, verify_experiment_identity


def parent_state(tmp_path, tc):
    mc = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16, bf16=False)
    params = {k: np.full(shape, .1, np.float32) for k, shape in expected_parameter_shapes(mc).items()}
    opt = {'step': np.array(7, np.int32),
           'm': {k: np.full_like(v, .01) for k, v in params.items()},
           'v': {k: np.full_like(v, .02) for k, v in params.items()}}
    replay = Replay(tc.replay_capacity)
    replay.add([(np.ones(6, np.uint8), np.zeros(21), np.zeros(16), i % 3,
                 np.zeros(30), 1., 7) for i in range(15)])
    log = tmp_path / 'old.jsonl'
    log.write_bytes(b'{"event":"old"}\n')
    meta = dict(model=asdict(mc), train=asdict(tc), source_lock={'engine': 'reference'},
                cycle=2, updates=7, frames=15, games=1, complete_samples=15, champion_cycle=1,
                collection_credit=12., total_seconds=10., numpy_rng=np.random.default_rng(42).bit_generator.state)
    store = Store(tmp_path / 'parent')
    path = store.save(params, opt, params, replay, meta, log)
    return mc, path, store.load_latest()


@pytest.mark.parametrize('remaining', [0, 1, 3, 4, 5])
@pytest.mark.parametrize('prefetch', [False, True])
@pytest.mark.parametrize('invalid_first', [False, True])
def test_exact_stop_runs_real_loop(tmp_path, monkeypatch, remaining, prefetch, invalid_first):
    from dougpu import train, model, inference
    tc = TrainConfig(engine='reference', backend='cpu', require_tpu=False, workers=1, envs_per_worker=3,
                     micro_batch=3, accumulation=1, replay_capacity=16, fresh_samples=12,
                     updates_per_cycle=4, sample_credit=True, replay_version_updates=True,
                     learner_prefetch=prefetch, eval_every=0, max_hours=1, target_updates=7+remaining)
    mc, _, parent = parent_state(tmp_path, tc)
    work = tmp_path / 'run'
    calls = []
    pools = []

    class Pool:
        requests = [None] * 3

        def __init__(self, *a, **kw):
            pools.append(self)

        def submit(self, *a):
            pass

        def receive(self):
            return [(np.ones(6, np.uint8), np.zeros(21), np.zeros(16), i % 3,
                     np.zeros(30), 1., 7) for i in range(12)], [0]

        def close(self):
            pass

    def step(p, opt, batch):
        valid = not (invalid_first and not calls)
        calls.append(int(opt['step']))
        return p, dict(opt, step=opt['step']+int(valid)), np.array([1. if valid else np.nan, 1., 1., 1., 1., float(valid)])

    monkeypatch.setattr(train, 'ActorPool', Pool)
    monkeypatch.setattr(model, 'make_train_step', lambda *a: step)
    monkeypatch.setattr(inference, 'Inference', lambda *a, **kw:
                        SimpleNamespace(choose=lambda *a, **kw: [0]*3, last_stats={}))
    spec = tmp_path / 'config.json'
    spec.write_text(json.dumps({'model': asdict(mc), 'train': asdict(tc)}))
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(spec), '--workdir', str(work),
                                    '--savedir', str(tmp_path / 'parent')])
    train.main()
    child = Store(work / 'checkpoints').load_latest()
    assert child['meta']['updates'] == int(child['optimizer']['step']) == 7+remaining
    assert len(calls) == remaining + int(invalid_first and remaining > 0)
    assert child['meta']['update_endpoint']['status'] == 'COMPLETE'
    assert child['meta']['update_endpoint']['stop_reason'] == 'target_updates'
    fresh = child['meta']['complete_samples'] - parent['meta']['complete_samples']
    assert child['meta']['collection_credit'] == 12 - 3*remaining + fresh
    assert fresh == (12 if remaining + int(invalid_first) > 4 else 0)
    if remaining == 0:
        assert not pools
        assert child['meta']['numpy_rng'] == parent['meta']['numpy_rng']
        assert child['meta']['cycle'] == 2


def test_experiment_fork_whitelist_identity_and_guards(tmp_path, monkeypatch):
    tc = TrainConfig(engine='reference', backend='cpu', require_tpu=False, micro_batch=64,
                     accumulation=4, history_groups=2, replay_capacity=16, sample_credit=True,
                     replay_version_updates=True, eval_every=0)
    mc, path, parent = parent_state(tmp_path, tc)
    new = replace(tc, accumulation=8, target_updates=12, max_hours=10)
    lock = {'engine': 'reference', 'doutpu_sha256': 'reviewed-new-code'}
    digest = sha256_file(path)
    config = lambda t: {'model': asdict(mc), 'train': asdict(t)}
    with pytest.raises(ValueError, match='Effective batch'):
        check_training_semantics(asdict(tc), new)
    with pytest.raises(ValueError, match='Effective batch'):
        fork(path.parent, tmp_path/'performance', '', config(new), lock)
    for field, value in [('lr', .001), ('replay_capacity', 32), ('grad_clip', 2.),
                         ('workers', 1), ('attention_impl', 'xla'), ('history_groups', 1),
                         ('sample_credit', False), ('save_replay', False), ('target_updates', None)]:
        with pytest.raises(ValueError):
            fork(path.parent, tmp_path/field, '', config(replace(new, **{field: value})), lock,
                 algorithm_experiment=True, parent_sha256=digest)
    with pytest.raises(ValueError, match='SHA256'):
        fork(path.parent, tmp_path/'badsha', '', config(new), lock,
             algorithm_experiment=True, parent_sha256='wrong')
    work = tmp_path/'experiment'
    fork(path.parent, work, '', config(new), lock, algorithm_experiment=True, parent_sha256=digest)
    child = Store(work/'checkpoints').load_latest()
    assert all(verify_experiment_identity(parent, child).values())
    experiment = child['meta']['algorithm_experiment']
    assert experiment['allowed_semantic_delta'] == {'accumulation': {'old': 4, 'new': 8}}
    assert (experiment['fork_updates'], experiment['fork_complete_samples'],
            experiment['fork_batch_size'], experiment['experiment_batch_size']) == (7, 15, 256, 512)
    assert child['meta']['reason'] == 'algorithm_experiment_fork'
    assert child['meta']['update_endpoint']['status'] == 'INCOMPLETE'
    check_training_semantics(child['meta']['train'], new)
    assert sha256_file(path) == digest
    for target in (None, 13):
        with pytest.raises(ValueError, match='registered'):
            fork(work/'checkpoints', tmp_path/f'retarget-{target}', '',
                 config(replace(new, target_updates=target)), lock)
    for component in ('params', 'champion', 'optimizer', 'replay'):
        from copy import deepcopy
        broken = deepcopy(child)
        values = broken[component]
        key = next(iter(values))
        if component == 'optimizer':
            key = 'step'
        values[key] = values[key] + 1
        with pytest.raises(ValueError, match='changed state'):
            verify_experiment_identity(parent, broken)
    from dougpu import train
    spec, lockfile = tmp_path/'resume.json', tmp_path/'lock.json'
    spec.write_text(json.dumps(config(replace(new, max_hours=1e-15))))
    lockfile.write_text(json.dumps(lock))
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(spec), '--workdir', str(work),
                                    '--source-lock', str(lockfile)])
    train.main()
    resumed = Store(work/'checkpoints').load_latest()
    assert resumed['meta']['algorithm_experiment'] == experiment
    assert resumed['meta']['updates'] == 7
    assert resumed['meta']['update_endpoint']['status'] == 'INCOMPLETE'
    spec.write_text(json.dumps(config(replace(new, target_updates=13))))
    with pytest.raises(ValueError, match='registered'):
        train.main()


@pytest.mark.parametrize('value', [-1, 1.5, True, '7'])
def test_target_validation(value):
    with pytest.raises(ValueError, match='target_updates'):
        TrainConfig(target_updates=value).validate()


@pytest.mark.parametrize('limit', ['time', 'cycle', 'below'])
def test_incomplete_or_invalid_endpoint(tmp_path, monkeypatch, limit):
    from dougpu import train
    tc = TrainConfig(engine='reference', backend='cpu', require_tpu=False, replay_capacity=16,
                     target_updates=6 if limit == 'below' else 12,
                     max_hours=1e-15 if limit == 'time' else 1, max_cycles=2)
    mc, _, _ = parent_state(tmp_path, tc)
    spec = tmp_path/'config.json'
    spec.write_text(json.dumps({'model': asdict(mc), 'train': asdict(tc)}))
    monkeypatch.setattr(train, 'ActorPool', lambda *a, **kw: SimpleNamespace(close=lambda: None))
    work = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(spec), '--workdir', str(work),
                                    '--savedir', str(tmp_path/'parent')])
    if limit == 'below':
        with pytest.raises(ValueError, match='below'):
            train.main()
    else:
        train.main()
        saved = Store(work/'checkpoints').load_latest()
        assert saved['meta']['updates'] == 7
        assert saved['meta']['update_endpoint']['status'] == 'INCOMPLETE'
        assert saved['meta']['update_endpoint']['stop_reason'] == limit+'_limit'


@pytest.mark.parametrize('failure', ['save', 'actor_close', 'signal'])
def test_termination_is_not_a_commit(tmp_path, monkeypatch, failure):
    from dougpu import train
    import signal
    tc = TrainConfig(engine='reference', backend='cpu', require_tpu=False, replay_capacity=16,
                     target_updates=7 if failure == 'save' else 12, max_cycles=3, max_hours=1)
    mc, _, _ = parent_state(tmp_path, tc)
    spec = tmp_path/'config.json'
    spec.write_text(json.dumps({'model': asdict(mc), 'train': asdict(tc)}))
    work = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(spec), '--workdir', str(work),
                                    '--savedir', str(tmp_path/'parent')])

    class Pool:
        def __init__(self, *args, **kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        def close(self):
            if failure == 'actor_close':
                raise RuntimeError('actor close failed')

    monkeypatch.setattr(train, 'ActorPool', Pool)
    original_save = Store.save

    def save(self, params, opt, champion, replay, meta, log):
        if failure == 'save' and meta['reason'] == 'session_end':
            raise OSError('final save failed')
        return original_save(self, params, opt, champion, replay, meta, log)

    monkeypatch.setattr(Store, 'save', save)
    if failure == 'signal':
        train.main()
    else:
        with pytest.raises((OSError, RuntimeError), match='failed'):
            train.main()
    saved = Store(work/'checkpoints').load_latest()
    assert saved['meta']['update_endpoint']['status'] == ('COMPLETE' if failure == 'save' else 'INCOMPLETE')
    events = [json.loads(line) for line in saved['log'].splitlines()]
    if failure == 'save':
        assert saved['meta']['reason'] == 'session_start'
        assert b'session_end' in (work/'metrics.jsonl').read_bytes()
        assert not any(row.get('event') == 'session_end' for row in events)
    elif failure == 'actor_close':
        assert saved['meta']['reason'] == 'error'
        assert saved['meta']['update_endpoint']['stop_reason'] == 'error'
    else:
        assert events[-1]['event'] == 'session_end'
        assert saved['meta']['update_endpoint']['stop_reason'] == 'signal'
