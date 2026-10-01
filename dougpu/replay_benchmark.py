"""Host-only sampler A/B on a verified replay; no training state is modified."""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from .checkpoint import Store, atomic_bytes, json_bytes, sha256_file
from .config import TrainConfig
from .replay import Replay, group_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--baseline-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phases', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=9)
    args = parser.parse_args()
    if args.phases < 1 or args.repeats < 1:
        parser.error('phases and repeats must be positive')
    cfg = TrainConfig(**json.loads(Path(args.config).read_text())['train']).validate()
    saved = Store(args.checkpoint_dir).load_latest()
    if saved is None or saved['replay'] is None:
        raise ValueError('A verified checkpoint containing replay is required')
    updates_unit = saved['meta']['train'].get('replay_version_updates', False)
    cfg = replace(cfg, replay_version_updates=updates_unit)
    version = saved['meta']['updates' if updates_unit else 'cycle']
    path = Path(args.baseline_source)/'dougpu/replay.py'
    if not path.is_file():
        path = Path(args.baseline_source)/'doutpu/replay.py'
    spec = importlib.util.spec_from_file_location('dougpu._previous_replay', path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    replays = {'baseline': old.Replay(cfg.replay_capacity), 'optimized': Replay(cfg.replay_capacity)}
    for replay in replays.values():
        replay.restore(saved['replay'])

    def prepare(name):
        replay = replays[name]
        if name == 'optimized':
            bins = replay.strata(cfg, version)
            assert all(len(bucket) for bucket in bins)
            return bins
        age = cfg.replay_max_age * (cfg.updates_per_cycle if cfg.replay_version_updates else 1)
        ids = replay.eligible(version, age)
        assert len(ids) and all(np.any(replay.data['role'][ids] == role) for role in range(3))
        return None

    def sample(name, rng, bins):
        replay = replays[name]
        batch = (replay.sample(cfg, rng, version, bins) if name == 'optimized'
                 else replay.sample(cfg, rng, version))
        return group_history(batch, cfg.history_groups) if cfg.history_groups > 1 else (batch,)

    rngs = {name: np.random.default_rng(20260929) for name in replays}
    for _ in range(40):
        bins = {name: prepare(name) for name in replays}
        for _ in range(cfg.updates_per_cycle):
            a = sample('baseline', rngs['baseline'], bins['baseline'])
            b = sample('optimized', rngs['optimized'], bins['optimized'])
            for x, y in zip(a, b):
                for key in x:
                    np.testing.assert_array_equal(x[key], y[key])
            assert rngs['baseline'].bit_generator.state == rngs['optimized'].bit_generator.state
    timings = {name: [] for name in replays}
    for repeat in range(args.repeats):
        order = list(replays) if repeat % 2 == 0 else list(replays)[::-1]
        for name in order:
            rng = np.random.default_rng(123456+repeat)
            tick = time.perf_counter()
            for _ in range(args.phases):
                bins = prepare(name)
                for _ in range(cfg.updates_per_cycle):
                    sample(name, rng, bins)
            timings[name].append((time.perf_counter()-tick)/args.phases)
    medians = {name: float(np.median(values)) for name, values in timings.items()}
    result = {'kind': 'host_sampler_including_readiness_and_history_grouping',
              'numpy': np.__version__, 'replay_size': replays['baseline'].size,
              'checkpoint_sha256': sha256_file(saved['path']),
              'batch_size': cfg.batch_size, 'updates_per_phase': cfg.updates_per_cycle,
              'history_groups': cfg.history_groups, 'parity_batches': 40*cfg.updates_per_cycle,
              'batch_and_rng_exact': True, 'phases_per_repeat': args.phases,
              'seconds_per_phase': timings, 'median_seconds_per_phase': medians,
              'latency_reduction': 1-medians['optimized']/medians['baseline'],
              'note': 'Host preparation only; excludes device transfer, learner, actors, evaluation and saves.'}
    atomic_bytes(args.output, json_bytes(result))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
