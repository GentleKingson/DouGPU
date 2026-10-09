"""CPU protocol acceptance: --train creates two real sessions; default is NumPy-only readback.

Run in a GPU-free container: python scripts/check_session_chain.py --train EVIDENCE_DIR
Then on either host: python scripts/check_session_chain.py EVIDENCE_DIR
The frozen intent precedes execution; receipt hashes/session IDs are filled only after saves.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dougpu.checkpoint import Store, copy_verified, json_bytes
from dougpu.config import ModelConfig, TrainConfig
from dougpu.files import sha256_file
from scripts.protocol_gate import audit_execution, require


def train_chain(root):
    import platform
    import jax
    import numpy as np
    import run_local

    require(jax.default_backend() == 'cpu', 'CPU required')
    root.mkdir(parents=True, exist_ok=False)
    work = root/'run'
    (work/'source').mkdir(parents=True)
    run_local.archive_trainer(work)
    lock = {'engine': 'reference', 'encoding_schema': 1}
    (work/'source/source_lock.json').write_bytes(json_bytes(lock))
    mc = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16, bf16=False)
    tc = TrainConfig(engine='reference', backend='cpu', require_tpu=False, workers=1,
                     envs_per_worker=1, infer_batch=1, action_chunk=64, micro_batch=3,
                     accumulation=1, replay_capacity=256, fresh_samples=3, updates_per_cycle=1,
                     max_cycles=100, max_hours=0.1, eval_every=0, checkpoint_seconds=3600,
                     keep_checkpoints=1, worker_timeout=60)
    versions = dict(python=platform.python_version(), jax=jax.__version__, numpy=np.__version__, backend='cpu')
    sessions = []
    for end in (1, 2):
        tc.target_updates = end
        sessions.append(dict(start_updates=end-1, end_updates=end, model=asdict(mc), train=asdict(tc),
            versions=versions, source_lock=lock, source_sha256=sha256_file(work/'source/trainer_source.zip')))
    intent = dict(scope='CPU protocol acceptance only; TRAINING_PAUSED; no NTP authorization',
                  GPU_hours=0, timeout_seconds_per_session=360, sessions=sessions)
    (root/'frozen-intent.json').write_bytes(json_bytes(intent))
    plan = dict(sessions=deepcopy(sessions), intent_sha256=sha256_file(root/'frozen-intent.json'))
    previous = None
    for index, session in enumerate(plan['sessions']):
        config = root/f'config-{index}.json'
        config.write_bytes(json_bytes({k: session[k] for k in ('model', 'train')}))
        with (root/f'session-{index}.stdout').open('w') as stdout:
            subprocess.run([sys.executable, '-m', 'dougpu.train', '--config', str(config),
                            '--workdir', str(work), '--source-lock', str(work/'source/source_lock.json')],
                           env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES=''),
                           stdout=stdout, stderr=subprocess.STDOUT, timeout=360, check=True)
        saved = Store(work/'checkpoints').load_latest()
        require(saved['meta']['updates'] == index+1, 'Wrong saved endpoint')
        destination = work/'endpoints'/saved['sha256']
        destination.mkdir(parents=True)
        path = Path(saved['path'])
        copy_verified(path, destination/path.name, saved['sha256'])
        marker = path.with_suffix('.ok.json')
        copy_verified(marker, destination/marker.name)
        require(Store(destination).load_latest()['sha256'] == saved['sha256'], 'Frozen endpoint mismatch')
        session.update(session_id=saved['meta']['session_id'], checkpoint_sha256=saved['sha256'],
                       input_checkpoint_sha256=previous)
        previous = saved['sha256']
    plan['checkpoint_sha256'] = previous
    (root/'execution-plan.json').write_bytes(json_bytes(plan))
    shutil.rmtree(work/'jax_cache', ignore_errors=True)


def check(root):
    plan = json.loads((root/'execution-plan.json').read_text())
    intent = json.loads((root/'frozen-intent.json').read_text())
    require(sha256_file(root/'frozen-intent.json') == plan['intent_sha256'], 'Frozen intent hash mismatch')
    require(len(plan['sessions']) == len(intent['sessions']), 'Frozen intent session count mismatch')
    for actual, frozen in zip(plan['sessions'], intent['sessions']):
        require({k: actual[k] for k in frozen} == frozen, 'Frozen intent session mismatch')
    work = root/'run'
    with redirect_stdout(io.StringIO()):
        positive = audit_execution(work, plan)
    failures = {}
    for case in ('missing_middle_end', 'middle_incomplete', 'forged_sha', 'missing_zip',
                 'continuity', 'marker', 'actor_seeds', 'source', 'runtime_version'):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'run'
            shutil.copytree(work, dest)
            altered = deepcopy(plan)
            first = altered['sessions'][0]
            endpoint = dest/'endpoints'/first['checkpoint_sha256']
            path = next(endpoint.glob('*.zip'))
            saved = Store(endpoint).load_latest()
            if case == 'forged_sha':
                first['checkpoint_sha256'] = '0'*64
            elif case == 'missing_zip':
                path.unlink()
            elif case == 'marker':
                path.with_suffix('.ok.json').write_text('{}')
            elif case == 'source':
                (dest/'source/trainer_source.zip').write_bytes(b'invalid')
            elif case == 'runtime_version':
                first['versions']['jax'] = 'unapproved'
            elif case == 'continuity':
                altered['sessions'][1]['start_updates'] += 1
            else:
                rows = [json.loads(line) for line in saved['log'].splitlines()]
                if case == 'missing_middle_end':
                    rows = [r for r in rows if r['event'] != 'session_end']
                elif case == 'middle_incomplete':
                    saved['meta']['update_endpoint']['status'] = 'INCOMPLETE'
                    rows[-1]['update_endpoint']['status'] = 'INCOMPLETE'
                else:
                    saved['meta']['actor_start']['worker_seeds'][0] += 1
                    next(r for r in rows if r['event'] == 'actor_start')['worker_seeds'][0] += 1
                log = Path(tmp)/'log.jsonl'
                log.write_bytes(b''.join(json.dumps(r).encode()+b'\n' for r in rows))
                forged_dir = Path(tmp)/'forged'
                from dougpu.replay import Replay
                replay = Replay(saved['meta']['train']['replay_capacity'])
                replay.restore(saved['replay'])
                forged = Store(forged_dir).save(saved['params'], saved['optimizer'], saved['champion'],
                                                replay, saved['meta'], log)
                sha = sha256_file(forged)
                shutil.copytree(forged_dir, dest/'endpoints'/sha)
                first['checkpoint_sha256'] = sha
                altered['sessions'][1]['input_checkpoint_sha256'] = sha
            try:
                with redirect_stdout(io.StringIO()):
                    audit_execution(dest, altered)
            except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
                failures[case] = str(exc)
            except __import__('zipfile').BadZipFile as exc:
                failures[case] = str(exc)
            else:
                raise AssertionError('Gate accepted ' + case)
    return dict(positive=positive, negatives=failures, plan_sha256=sha256_file(root/'execution-plan.json'),
                intent_sha256=plan['intent_sha256'], GPU_work=0, training='PAUSED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    if args.train:
        train_chain(args.root.resolve())
    print(json.dumps(check(args.root), indent=2))


if __name__ == '__main__':
    main()
