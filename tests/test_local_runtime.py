from dataclasses import asdict, replace
import json
from types import SimpleNamespace
import numpy as np
import pytest

from dougpu.config import ModelConfig, TrainConfig
from dougpu.runtime import configure_runtime, requested_backend, verify_backend
from dougpu.semantics import check_training_semantics, check_array_state, expected_parameter_shapes
from dougpu.checkpoint import Store, sha256_file
from dougpu.replay import Replay
from fork_run import fork


def test_explicit_cuda_overrides_inherited_cpu_mode(monkeypatch, tmp_path):
    monkeypatch.setattr('dougpu.runtime.sys.platform', 'linux')
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    monkeypatch.setenv('JAX_COMPILATION_CACHE_DIR', str(tmp_path/'cache'))
    config = TrainConfig(backend='cuda', require_tpu=False)
    assert configure_runtime(config, tmp_path/'other') == 'cuda'
    import os
    assert os.environ['JAX_PLATFORMS'] == 'cuda'
    assert requested_backend(TrainConfig(require_tpu=False)) == 'cpu'
    assert requested_backend(TrainConfig()) == 'tpu'
    fake_cpu = SimpleNamespace(default_backend=lambda: 'cpu', devices=lambda: ['CPU'])
    with pytest.raises(RuntimeError, match='Refusing'):
        verify_backend(config, fake_cpu)
    fake_gpu = SimpleNamespace(default_backend=lambda: 'gpu',
        devices=lambda: [SimpleNamespace(platform='gpu'), SimpleNamespace(platform='gpu')])
    with pytest.raises(RuntimeError, match='exactly one'):
        verify_backend(config, fake_gpu)
    def unavailable():
        raise AssertionError()
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        verify_backend(config, SimpleNamespace(default_backend=unavailable))


def test_performance_changes_keep_learning_batch_and_objective():
    before = TrainConfig(sample_credit=True, replay_version_updates=True)
    candidate = replace(before, micro_batch=64, accumulation=4, workers=8,
                        infer_batch=32, backend='cuda', require_tpu=False, learner_remat=False)
    check_training_semantics(asdict(before), candidate)
    for key, value in [('micro_batch', 128), ('fresh_samples', 1024), ('updates_per_cycle', 8),
                       ('ntp_weight', 0.), ('replay_max_age', 128)]:
        with pytest.raises(ValueError):
            check_training_semantics(asdict(before), replace(candidate, **{key: value}))


def make_state(tmp_path):
    model = ModelConfig(width=32, layers=1, heads=2, ffn=64, q_hidden=32, bf16=False)
    config = TrainConfig(require_tpu=False, backend='cpu', engine='reference',
                         replay_capacity=12, sample_credit=True, replay_version_updates=True)
    rng = np.random.default_rng(99)
    params = {k: rng.normal(size=s).astype(np.float32)*.01 for k, s in expected_parameter_shapes(model).items()}
    opt = {'step': np.array(12, np.int32), 'm': {k: np.ones_like(v)*.1 for k, v in params.items()},
           'v': {k: np.ones_like(v)*.2 for k, v in params.items()}}
    champion = {k: v*.7 for k, v in params.items()}
    replay = Replay(12)
    replay.add([(np.ones(6, np.uint8), np.zeros(21), np.zeros(16), i%3, np.zeros(30),
                 float(1 if i%3 == 0 else -1), 8+i//3) for i in range(9)])
    lock = {'engine': 'reference', 'encoding_schema': 1, 'doutpu_sha256': 'old'}
    meta = {'model': asdict(model), 'train': asdict(config), 'source_lock': lock,
            'cycle': 3, 'updates': 12, 'champion_cycle': 2, 'frames': 3072,
            'games': 50, 'complete_samples': 2700, 'collection_credit': 73.,
            'numpy_rng': rng.bit_generator.state}
    path = Store(tmp_path/'source').save(params, opt, champion, replay, meta)
    return model, config, lock, path


def test_local_migration_preserves_full_state_and_source(tmp_path):
    model, config, lock, path = make_state(tmp_path)
    source = Store(tmp_path/'source').load_latest()
    check_array_state(source, model)
    before = sha256_file(path)
    new = replace(config, micro_batch=64, accumulation=4, backend='cuda')
    target_lock = dict(lock, doutpu_sha256='new')
    fork(tmp_path/'source', tmp_path/'target', None,
         {'model': asdict(model), 'train': asdict(new)}, target_lock)
    migrated = Store(tmp_path/'target'/'checkpoints').load_latest()
    for group in ('params', 'champion', 'replay'):
        for key in source[group]:
            np.testing.assert_array_equal(source[group][key], migrated[group][key])
    for group in ('m', 'v'):
        for key in source['optimizer'][group]:
            np.testing.assert_array_equal(source['optimizer'][group][key], migrated['optimizer'][group][key])
    for key in ('cycle', 'updates', 'champion_cycle', 'frames', 'games', 'complete_samples', 'numpy_rng', 'collection_credit'):
        assert source['meta'][key] == migrated['meta'][key]
    assert migrated['meta']['source_lock'] == target_lock
    assert sha256_file(path) == before
    with pytest.raises(ValueError, match='already has'):
        fork(tmp_path/'source', tmp_path/'target', None,
             {'model': asdict(model), 'train': asdict(new)}, target_lock)


def test_fork_preserves_historical_selection_seeds(tmp_path):
    model, config, lock, _ = make_state(tmp_path)
    candidate = replace(config, eval_seed=900002)
    fork(tmp_path/'source', tmp_path/'target', None,
         {'model': asdict(model), 'train': asdict(candidate)}, lock)
    saved = Store(tmp_path/'target'/'checkpoints').load_latest()
    assert saved['meta']['selection_seeds'] == [900001, 900002]


def test_policy_only_or_inconsistent_adam_state_rejected(tmp_path):
    model, _, _, _ = make_state(tmp_path)
    saved = Store(tmp_path/'source').load_latest()
    saved['optimizer']['step'] = np.array(0, np.int32)
    with pytest.raises(ValueError, match='Adam step'):
        check_array_state(saved, model)
    saved = Store(tmp_path/'source').load_latest()
    del saved['params']['ntp']
    with pytest.raises(ValueError, match='auxiliary'):
        check_array_state(saved, model)


def test_run_lock_rejects_concurrent_owner(tmp_path):
    from run_local import run_lock
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match='already in use'):
            with run_lock(tmp_path):
                pass


def test_runtime_config_rejects_alternate_attention_with_research_kv():
    with pytest.raises(ValueError, match='KV'):
        TrainConfig(backend='cuda', attention_impl='cudnn', selfplay_kv_cache=True).validate()


def test_additional_cycles_use_newer_verified_mirror(tmp_path, monkeypatch):
    import run_local
    model, config, lock, path = make_state(tmp_path)
    saved = Store(tmp_path/'source').load_latest()
    run, mirror = tmp_path/'run', tmp_path/'mirror'
    replay = Replay(config.replay_capacity)
    replay.restore(saved['replay'])
    for destination, cycle in ((run/'checkpoints', 3), (mirror, 5)):
        Store(destination).save(saved['params'], saved['optimizer'], saved['champion'], replay,
                                dict(saved['meta'], cycle=cycle))
    spec = {'model': asdict(model), 'train': asdict(config)}
    monkeypatch.setattr(run_local, 'prepared', lambda _: (spec, model, config, lock))
    called = []
    monkeypatch.setattr(run_local, 'invoke', lambda module, argv: called.append((module, argv)))
    monkeypatch.setattr('sys.argv', ['run_local.py', 'train', '--run-dir', str(run),
                                    '--mirror-dir', str(mirror), '--cycles', '2'])
    run_local.main()
    actual = json.loads((run/'session_config.json').read_text())
    assert actual['train']['max_cycles'] == 7
    assert called[0][0] == 'dougpu.train'
    assert str(mirror) in [str(v) for v in called[0][1]]
