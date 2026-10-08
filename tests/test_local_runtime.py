from dataclasses import asdict, replace
import json
import subprocess
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


def test_full_mirror_survives_workdir_deletion_and_rejects_corruption(tmp_path):
    import shutil
    import zipfile
    from pathlib import Path
    from dougpu.checkpoint import load_policy
    from fork_run import verify_experiment_identity

    model, config, _, _ = make_state(tmp_path)
    original = Store(tmp_path/'source').load_latest()
    replay = Replay(config.replay_capacity)
    replay.restore(original['replay'])
    run, mirror = tmp_path/'run', tmp_path/'mirror'
    run.mkdir()
    log = run/'metrics.jsonl'
    log.write_text('{"event":"train","updates":12,"successful_steps":4}\n')
    meta = dict(original['meta'], total_seconds=1.,
                versions={'git_commit': 'test-source', 'git_dirty': False})
    store = Store(run/'checkpoints', mirror)
    archive = store.save(original['params'], original['optimizer'], original['champion'],
                         replay, meta, log)
    assert store.last_remote_ok is True
    before = store.load_latest()
    hashes = {p.name: sha256_file(p) for p in store.local.iterdir()}
    assert hashes == {p.name: sha256_file(mirror/p.name) for p in store.local.iterdir()}
    assert (mirror/'metrics.jsonl').read_bytes() == before['log']
    shutil.rmtree(run)
    shutil.rmtree(tmp_path/'source')

    recovered = Store(run/'checkpoints', mirror).load_latest()
    assert Path(recovered['path']) == mirror/archive.name
    check_array_state(recovered, model)
    assert all(verify_experiment_identity(before, recovered).values())
    assert recovered['meta'] == before['meta']
    restored_replay = Replay(config.replay_capacity)
    restored_replay.restore(recovered['replay'])
    for key, value in replay.export().items():
        np.testing.assert_array_equal(restored_replay.export()[key], value)
    rng = np.random.default_rng()
    rng.bit_generator.state = recovered['meta']['numpy_rng']
    assert rng.bit_generator.state == before['meta']['numpy_rng']
    for name, group in (('latest', 'params'), ('best', 'champion')):
        policy, _ = load_policy(mirror/f'{name}_policy.npz')
        for key, value in policy.items():
            np.testing.assert_array_equal(value, recovered[group][key])
    assert hashes == {name: sha256_file(mirror/name) for name in hashes}

    # Validate both checksum layers, even if the outer marker matches a bad ZIP.
    archive = mirror/archive.name
    intact = archive.read_bytes()
    archive.write_bytes(b'broken archive')
    with pytest.raises(RuntimeError, match='not restarting silently'):
        Store(run/'checkpoints', mirror).load_latest()
    archive.write_bytes(intact)
    with zipfile.ZipFile(archive) as z:
        entries = {name: z.read(name) for name in z.namelist()}
    entries['metrics.jsonl'] += b'changed without updating manifest\n'
    with zipfile.ZipFile(archive, 'w') as z:
        for name, content in entries.items():
            z.writestr(name, content)
    marker = archive.with_suffix('.ok.json')
    document = json.loads(marker.read_text())
    document['sha256'] = sha256_file(archive)
    marker.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match='not restarting silently'):
        Store(run/'checkpoints', mirror).load_latest()


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


@pytest.mark.parametrize('status,expected', [('', False), (' M dougpu/train.py\n', True)])
def test_git_provenance_records_clean_and_dirty(monkeypatch, status, expected):
    from dougpu.runtime import git_source
    monkeypatch.setattr('dougpu.runtime.subprocess.run', lambda args, **kw:
                        SimpleNamespace(stdout='.' if '--show-toplevel' in args else
                                        status if 'status' in args else 'abc123\n'))
    assert git_source('.') == {'git_commit': 'abc123', 'git_dirty': expected}


@pytest.mark.parametrize('error', [FileNotFoundError(), subprocess.CalledProcessError(128, 'git'),
                                  subprocess.TimeoutExpired('git', 5)])
def test_git_provenance_unknown_is_not_clean(monkeypatch, error):
    from dougpu.runtime import git_source
    def unavailable(*args, **kwargs):
        raise error
    monkeypatch.setattr('dougpu.runtime.subprocess.run', unavailable)
    assert git_source('.') == {'git_commit': 'unknown', 'git_dirty': 'unknown'}


@pytest.mark.parametrize('mode,plugin,error', [
    ('cpu', 'jax-cuda12-plugin', 'CPU-only'), ('cpu', 'jax_cuda13_plugin', 'CPU-only'),
    ('gpu', 'jax-cuda12-plugin', 'CUDA 13 install'), ('gpu', 'jax-cuda13-plugin', None),
    ('cpu', '', None), ('gpu', '', None),
])
def test_installer_dependency_scan(monkeypatch, mode, plugin, error):
    from pathlib import Path
    script = (Path(__file__).resolve().parents[1]/'scripts/install.sh').read_text()
    scan = script.split("<<'PYTHON'\n", 1)[1].split('\nPYTHON', 1)[0]
    monkeypatch.setattr('sys.argv', ['-', mode])
    monkeypatch.setattr('importlib.metadata.distributions', lambda: [SimpleNamespace(metadata={'Name': plugin})])
    if error:
        with pytest.raises(SystemExit, match=error):
            exec(scan, {})
    else:
        exec(scan, {})


def test_git_provenance_does_not_attribute_parent_repository(tmp_path):
    from dougpu.runtime import git_source
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test',
                    '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false',
                    'commit', '--allow-empty', '-m', 'test'], check=True, capture_output=True)
    child = tmp_path/'DouGPU'
    child.mkdir()
    assert git_source(tmp_path)['git_commit'] != 'unknown'
    assert git_source(child) == {'git_commit': 'unknown', 'git_dirty': 'unknown'}
