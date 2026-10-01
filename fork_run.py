"""Copy a verified original checkpoint into a separate performance experiment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from dougpu.checkpoint import Store, atomic_bytes, json_bytes, sha256_file
from dougpu.config import ModelConfig, TrainConfig
from dougpu.replay import Replay
from dougpu.semantics import check_training_semantics


def fork(source, work, savedir, config, lock):
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
    path = target.save(saved['params'],saved['optimizer'],saved['champion'],replay,
                       newmeta,work/'metrics.jsonl')
    if savedir and target.last_remote_ok is not True:
        raise IOError('Fork checkpoint was not verified in the target savedir')
    atomic_bytes((savedir.parent if savedir else work)/'performance_provenance.json',json_bytes(provenance))
    print('FORK_VERIFIED',path.name,'source remains unchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--workdir',required=True)
    p.add_argument('--savedir',default='')
    p.add_argument('--config',required=True)
    p.add_argument('--source-lock',required=True)
    a=p.parse_args()
    fork(a.source,a.workdir,a.savedir,json.loads(Path(a.config).read_text()),
         json.loads(Path(a.source_lock).read_text()))
