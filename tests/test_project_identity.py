import argparse
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import run_local
from bootstrap import project_hash


def test_dougpu_identity_source_lock_and_archive(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    assert importlib.util.find_spec('dougpu.train') is not None
    assert importlib.util.find_spec('doutpu') is None
    lock = json.loads((root / 'source_lock.json').read_text())
    assert lock['doutpu_sha256'] == project_hash(root)

    project = tmp_path / 'DouGPU'
    package = project / 'dougpu'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('"""DouGPU test source."""\n')
    monkeypatch.setattr(run_local, 'ROOT', project)
    run = tmp_path / 'run'
    run_local.prepare_run(argparse.Namespace(
        run_dir=run, config=root / 'configs/cpu_smoke.json', upstream_cache=''))
    info = json.loads((run / 'run_info.json').read_text())
    assert info['project'] == 'DouGPU'
    assert info['source_notebook'] == 'DouTPU_v6e1_Optimized_Final.ipynb'
    prepared_lock = json.loads((run / 'source/source_lock.json').read_text())
    assert prepared_lock['doutpu_sha256'] == project_hash(project)
    with ZipFile(run / 'source/trainer_source.zip') as archive:
        assert 'dougpu/__init__.py' in archive.namelist()
        assert not any(name.startswith('doutpu/') for name in archive.namelist())


def test_replay_benchmark_accepts_current_and_original_packages(tmp_path, monkeypatch):
    import sys
    import numpy as np
    from dougpu.checkpoint import Store
    from dougpu.config import TrainConfig
    from dougpu.replay import Replay
    from dougpu.replay_benchmark import main

    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / 'configs/cpu_smoke.json').read_text())
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(spec))
    cfg = TrainConfig(**spec['train'])
    replay = Replay(cfg.replay_capacity)
    replay.add([(np.ones(6, np.uint8), np.zeros(21), np.zeros(16), role,
                 np.zeros(30), 1., 0) for role in range(3)])
    params = {'w': np.zeros(1, np.float32)}
    optimizer = {'step': np.array(0, np.int32), 'm': params, 'v': params}
    state = tmp_path / 'state'
    Store(state).save(params, optimizer, params, replay,
                     dict(spec, cycle=0, updates=0, champion_cycle=0))
    for name in ('dougpu', 'doutpu'):
        baseline = tmp_path / name
        (baseline / name).mkdir(parents=True)
        (baseline / name / 'replay.py').write_bytes((root / 'dougpu/replay.py').read_bytes())
        output = tmp_path / (name + '.json')
        monkeypatch.setattr(sys, 'argv', ['benchmark', '--config', str(config),
            '--checkpoint-dir', str(state), '--baseline-source', str(baseline),
            '--output', str(output), '--phases', '1', '--repeats', '1'])
        main()
        assert json.loads(output.read_text())['batch_and_rng_exact'] is True
