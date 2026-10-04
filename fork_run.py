"""Fork a verified checkpoint; algorithm changes require the explicit experiment mode."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from dougpu.checkpoint import Store, atomic_bytes, json_bytes, sha256_file
from dougpu.config import ModelConfig, TrainConfig
from dougpu.replay import Replay
from dougpu.semantics import check_training_semantics, check_array_state


def verify_experiment_identity(parent, child):
    """Compare round-tripped arrays byte-for-byte, including replay ring metadata."""
    import numpy as np

    def equal(a, b):
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

    checks = {key: equal(parent[key], child[key]) for key in ('params', 'optimizer', 'champion', 'replay')}
    for key in ('numpy_rng', 'collection_credit', 'cycle', 'updates', 'frames', 'games',
                'complete_samples', 'champion_cycle', 'total_seconds'):
        checks[key] = parent['meta'][key] == child['meta'][key]
    checks['metrics_log'] = parent['log'] == child['log']
    if not all(checks.values()):
        raise ValueError('Experiment fork changed state: ' + ', '.join(k for k, ok in checks.items() if not ok))
    return checks


def fork(source, work, savedir, config, lock, *, algorithm_experiment=False, parent_sha256=None):
    source,work = (Path(p).resolve() for p in (source,work))
    savedir = Path(savedir).resolve() if savedir else None
    for target in (work, savedir) if savedir else (work,):
        if source==target or source in target.parents or target in source.parents:
            raise ValueError('Source and target directories must be separate')
    mc,tc = ModelConfig(**config['model']).validate(),TrainConfig(**config['train']).validate()
    target = Store(work/'checkpoints',savedir,tc.keep_checkpoints)
    if target.load_latest() is not None:
        raise ValueError('Target already has checkpoints; resume it instead of forking again')
    saved = Store(work/'source_readonly',source).load_latest()
    if saved is None:
        raise FileNotFoundError('No verified source checkpoint; refusing to train from scratch')
    meta = saved['meta']
    if meta['model']!=asdict(mc):
        raise ValueError('Model configuration must be unchanged')
    if meta.get('algorithm_experiment') and tc.target_updates != meta['algorithm_experiment']['target_updates']:
        raise ValueError('Cannot change the registered algorithm experiment endpoint when forking')
    if algorithm_experiment:
        if not parent_sha256 or sha256_file(saved['path']) != parent_sha256:
            raise ValueError('Algorithm experiment requires the exact parent checkpoint SHA256')
        if tc.engine == 'douzero':
            from bootstrap import verify_local_source
            verify_local_source(Path(__file__).resolve().parent, lock)
        old_tc = TrainConfig(**meta['train']).validate()
        if (old_tc.micro_batch, old_tc.accumulation, tc.micro_batch, tc.accumulation) != (64, 4, 64, 8):
            raise ValueError('Only micro_batch=64, accumulation 4 -> 8 is allowed')
        # A distinct fork, never a bypass in the normal resume/performance guard.
        check_training_semantics(meta['train'], replace(tc, accumulation=4))
        for key, value in asdict(old_tc).items():
            if key not in ('accumulation', 'target_updates', 'max_hours', 'max_cycles') and value != getattr(tc, key):
                raise ValueError('Algorithm experiment changed non-whitelisted field: ' + key)
        if (not tc.resume or not tc.save_replay or not tc.sample_credit or not tc.replay_version_updates
                or tc.target_updates is None or tc.target_updates <= meta['updates']):
            raise ValueError('Experiment requires full-state resume, sample credits, update versions and a future target')
        if saved['replay'] is None or meta.get('algorithm_experiment'):
            raise ValueError('Experiment requires a full replay and an unforked algorithm parent')
        check_array_state(saved, mc)
        # Validate that the saved RNG can actually be restored before committing a fork.
        import numpy as np
        rng = np.random.default_rng()
        rng.bit_generator.state = meta['numpy_rng']
    else:
        check_training_semantics(meta['train'], tc)
    if meta['train'].get('sample_credit', False) and not tc.sample_credit:
        raise ValueError('Cannot disable saved sample credits during a performance migration')
    old_lock = meta['source_lock']
    for key in ('engine','encoding_schema','douzero_commit','upstream_files'):
        if old_lock.get(key)!=lock.get(key):
            raise ValueError('Rule engine or encoding schema changed: '+key)
    provenance = {'source_checkpoint':saved['path'], 'source_sha256':sha256_file(saved['path']),
                  'source_lock':old_lock, 'target_source_lock':lock,
                  'changes':'hardware/execution migration; no learning state reset',
                  'cycle':meta['cycle'], 'updates':meta['updates']}
    replay = None
    if saved['replay'] is not None:
        replay = Replay(tc.replay_capacity)
        replay.restore(saved['replay'])
    old_updates = meta['train'].get('replay_version_updates', False)
    if old_updates != tc.replay_version_updates:
        if old_updates or not tc.replay_version_updates:
            raise ValueError('Cannot convert update versions back to cycles')
        rows = [json.loads(line) for line in saved['log'].splitlines()]
        train_rows = [r for r in rows if r.get('event') == 'train']
        steps = meta['train']['updates_per_cycle']
        if (steps != tc.updates_per_cycle or meta['updates'] != meta['cycle']*steps or
                len(train_rows) != meta['cycle'] or any(r['successful_steps'] != steps for r in train_rows)):
            raise ValueError('Cannot infer historical behavior updates from this checkpoint')
        if replay is not None:
            import numpy as np
            converted = replay.data['version'][:replay.size].astype(np.int64)*steps
            if np.any(converted > np.iinfo(np.int32).max):
                raise ValueError('Behavior version exceeds replay storage')
            replay.data['version'][:replay.size] = converted
        provenance['replay_version_conversion'] = 'cycle * verified constant successful updates per cycle'
    atomic_bytes(work/'metrics.jsonl',saved['log'])
    selection_seeds = sorted(set(meta.get('selection_seeds', [])) | {meta['train']['eval_seed'], tc.eval_seed})
    newmeta = dict(meta,train=asdict(tc),source_lock=lock,reason='performance_fork',forked_from=provenance,
                   selection_seeds=selection_seeds)
    if algorithm_experiment:
        identity = verify_experiment_identity(saved, dict(saved, replay=replay.export()))
        provenance.update(changes='effective-batch / sample-row-dose experiment',
                          old_resolved_config={'model': meta['model'], 'train': asdict(old_tc)},
                          new_resolved_config={'model': asdict(mc), 'train': asdict(tc)},
                          allowed_semantic_delta={'accumulation': {'old': 4, 'new': 8}},
                          identity_checks=identity,
                          fork_updates=meta['updates'], fork_complete_samples=meta['complete_samples'],
                          fork_batch_size=256, experiment_batch_size=512,
                          target_updates=tc.target_updates,
                          legacy_cumulative_sample_update_ratio='not valid across mixed-batch fork')
        newmeta.update(reason='algorithm_experiment_fork', algorithm_experiment=provenance,
                       update_endpoint={'target_updates': tc.target_updates, 'status': 'INCOMPLETE',
                                        'stop_reason': 'not_started'})
    path = target.save(saved['params'],saved['optimizer'],saved['champion'],replay,
                       newmeta,work/'metrics.jsonl')
    if algorithm_experiment:
        child = target.load_latest()
        if child is None or Path(child['path']).name != path.name:
            raise IOError('Could not read back the exact experiment fork')
        verify_experiment_identity(saved, child)
        if child['meta'] != dict(newmeta, schema=meta['schema']):
            raise ValueError('Experiment fork metadata round-trip failed')
        if sha256_file(saved['path']) != parent_sha256:
            raise ValueError('Parent checkpoint changed while creating the experiment fork')
    if savedir and target.last_remote_ok is not True:
        raise IOError('Fork checkpoint was not verified in the target savedir')
    name = 'algorithm_experiment_provenance.json' if algorithm_experiment else 'performance_provenance.json'
    atomic_bytes((savedir.parent if savedir else work)/name,json_bytes(provenance))
    print('FORK_VERIFIED',path.name,'source remains unchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--workdir',required=True)
    p.add_argument('--savedir',default='')
    p.add_argument('--config',required=True)
    p.add_argument('--source-lock',required=True)
    p.add_argument('--algorithm-experiment', action='store_true')
    p.add_argument('--parent-sha256', help='Required exact parent identity for algorithm experiments')
    a=p.parse_args()
    fork(a.source,a.workdir,a.savedir,json.loads(Path(a.config).read_text()),
         json.loads(Path(a.source_lock).read_text()), algorithm_experiment=a.algorithm_experiment,
         parent_sha256=a.parent_sha256)
