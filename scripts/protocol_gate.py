"""Offline gate for the archived 20261008 baseline; never launches training.

Usage: python scripts/protocol_gate.py BASELINE_DIRECTORY FROZEN_PLAN.json
JSON receipt goes to stdout; only a complete joint audit exits zero.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dougpu.config import ModelConfig, TrainConfig
from dougpu.semantics import FIXED_FIELDS, check_training_semantics

STAGES = ('smoke', 'stage2000', 'stage20000')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def boundaries(rows):
    """Reuse audit-restarts.py's successful-step arithmetic; reject interleaving."""
    sessions = []
    seen = set()
    for row in rows:
        sid = row['session_id']
        if not sessions or sessions[-1]['session_id'] != sid:
            require(sid and sid not in seen, 'Repeated/interleaved session')
            seen.add(sid)
            start = sessions[-1]['end_updates'] if sessions else 0
            sessions.append(dict(session_id=sid, start_updates=start, end_updates=start))
        session = sessions[-1]
        require(type(row['updates']) is int, 'Invalid update count')
        if row['event'] == 'train':
            steps = row['successful_steps']
            require(type(steps) is int and steps >= 0, 'Invalid successful steps')
            require(row['updates'] - steps == session['end_updates'], 'Gap/duplicate train updates')
            session['end_updates'] = row['updates']
        else:
            require(row['updates'] == session['end_updates'], 'Event/update disagreement')
    require(bool(sessions), 'No sessions')
    return sessions


def category(key):
    if key in FIXED_FIELDS or key in ('micro_batch', 'accumulation'):
        return 'learning'
    if key in ('log_every', 'keep_checkpoints'):
        return 'recording'
    if key in ('backend', 'require_tpu', 'attention_impl', 'worker_timeout', 'max_hours'):
        return 'execution'
    return 'data_trajectory'


def compare(actual, plan):
    require(plan['allowed_difference'] == {'ntp_weight': {'A': .02, 'B': 0}},
            'Only the declared NTP difference is supported')
    require(len(actual) == len(plan['sessions']), 'Session count/boundary mismatch')
    differences = []
    for a, b in zip(actual, plan['sessions']):
        for key in ('start_updates', 'end_updates', 'stop_reason', 'initialization',
                    'model', 'versions', 'source_sha256', 'source_lock'):
            require(a[key] == b[key], f'{a["session_id"]}: mismatch: {key}')
        for cfg in (a, b):
            require(set(cfg['model']) == {f.name for f in fields(ModelConfig)}, 'Incomplete model config')
            require(set(cfg['train']) == {f.name for f in fields(TrainConfig)}, 'Incomplete train config')
            ModelConfig(**cfg['model']).validate()
            TrainConfig(**cfg['train']).validate()
        require(a['train']['ntp_weight'] == .02 and b['train']['ntp_weight'] == 0,
                'NTP treatment mismatch')
        candidate = dict(b['train'], ntp_weight=a['train']['ntp_weight'])
        check_training_semantics(a['train'], TrainConfig(**candidate))
        for key in a['train']:
            if a['train'][key] != b['train'][key]:
                differences.append(dict(session_id=a['session_id'], field=key,
                                        category=category(key), A=a['train'][key], B=b['train'][key]))
                require(key == 'ntp_weight', f'Undeclared {category(key)} difference: {key}')
    return differences


def read_stage(path, stage, previous):
    # Only checkpoint files are materialized, into an isolated temporary directory.
    import numpy as np
    from dougpu.checkpoint import Store
    from dougpu.replay import Replay
    from dougpu.semantics import check_array_state

    # ponytail: legacy archives cannot supply runtime attestations; no inference substitutes for evidence.
    with zipfile.ZipFile(path) as z:
        require(len(z.namelist()) == len(set(z.namelist())), 'Duplicate ZIP members')
        manifest = json.loads(z.read('evidence-manifest.json'))
        require(set(z.namelist()) == set(manifest) | {'evidence-manifest.json'}, 'Incomplete manifest')
        for name, sha in manifest.items():
            require(digest(z.read(name)) == sha, 'Manifest mismatch: ' + name)
        read = lambda name: json.loads(z.read(name))
        expected = read(stage + '-verification.json')
        cfg = read('run/resolved_config.json')
        require(cfg == read('run/session_config.json'), 'Session/resolved config mismatch')
        exit_record = read(stage + '-exit.json')
        require(exit_record['stage'] == stage and exit_record['returncode'] == 0
                and exit_record['timed_out'] is False, 'Unsuccessful process exit')
        with tempfile.TemporaryDirectory() as tmp:
            for name in manifest:
                if name.startswith('run/checkpoints/ckpt_'):
                    require(len(Path(name).parts) == 3, 'Invalid checkpoint path')
                    (Path(tmp) / Path(name).name).write_bytes(z.read(name))
            saved = Store(tmp).load_latest()
            require(saved is not None, 'Missing full checkpoint')
            checkpoint = Path(saved['path']).name
            sha = digest(Path(saved['path']).read_bytes())
        require(sha == expected['checkpoint_sha256'], 'Wrong checkpoint / fallback')
        require(saved['meta'] == expected['meta'], 'Checkpoint metadata mismatch')
        require(digest(saved['log']) == expected['log_sha256'], 'Checkpoint log mismatch')
        meta = saved['meta']
        require(cfg == {'model': meta['model'], 'train': meta['train']}, 'Actual config contradicts checkpoint')
        check_array_state(saved, ModelConfig(**meta['model']))
        replay = Replay(meta['train']['replay_capacity'])
        replay.restore(saved['replay'])
        rng = np.random.default_rng()
        rng.bit_generator.state = meta['numpy_rng']
        raw = z.read('run/metrics.jsonl')
        require(raw.startswith(saved['log']), 'Checkpoint log is not a prefix of external log')
        rows = [json.loads(line) for line in raw.splitlines()]
        sessions = boundaries(rows)
        current = sessions[-1]
        own = [r for r in rows if r['session_id'] == current['session_id']]
        starts = [r for r in own if r['event'] == 'start']
        ends = [r for r in own if r['event'] == 'session_end']
        require(len(starts) == len(ends) == 1, 'Missing/duplicate start or end event')
        require(starts[0]['updates'] == current['start_updates'], 'Start boundary mismatch')
        require(own[-1] == ends[0] and meta['updates'] == current['end_updates'], 'End boundary mismatch')
        require(meta['reason'] == 'session_end' and ends[0]['stop_reason'] == 'target_updates', 'Incomplete endpoint')
        require(meta['update_endpoint'] == ends[0]['update_endpoint'] == {
            'target_updates': current['end_updates'], 'status': 'COMPLETE', 'stop_reason': 'target_updates'},
            'Endpoint metadata mismatch')
        require(starts[0]['versions'] == meta['versions'] and starts[0]['source_lock'] == meta['source_lock'],
                'Execution identity mismatch')
        require(exit_record['ended'] >= ends[0]['wall_time'], 'Exit precedes session end')
        log = z.read(stage + '.log').decode()
        stdout_events = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
        require(starts[0] in stdout_events and ends[0] in stdout_events, 'Missing independent stdout endpoints')
        require(any(r['event'] == 'checkpoint' and r.get('reason') == 'session_end' for r in own),
                'Missing final checkpoint event')
        resumes = re.findall(r'^\[RESUME\] \S*/(ckpt_\S+\.zip) cycle=\d+, updates=(\d+),', log, re.M)
        if previous:
            require(resumes == [(previous['checkpoint'], str(previous['end_updates']))], 'Wrong/missing resume source')
            require(sessions[:-1] == previous['history'], 'Restored history mismatch')
        else:
            require(not resumes and len(sessions) == 1, 'First session must start from zero')
        source_sha = digest(z.read('source.zip'))
        require(source_sha == read('protocol.json')['source_sha256'], 'Frozen source mismatch')
        return dict(current, history=sessions, checkpoint=checkpoint, checkpoint_sha256=sha,
                    input_checkpoint_sha256=previous['checkpoint_sha256'] if previous else None,
                    resume_evidence='stdout filename + prior archive hash; no runtime input digest',
                    model=meta['model'], train=meta['train'], versions=meta['versions'],
                    source_lock=meta['source_lock'], source_sha256=source_sha,
                    stop_reason=ends[0]['stop_reason'], initialization='resume_previous' if previous else 'fresh',
                    adam_step=int(saved['optimizer']['step']),
                    checkpoint_omits_end=not any(json.loads(x)['event'] == 'session_end'
                        and json.loads(x)['session_id'] == current['session_id'] for x in saved['log'].splitlines()),
                    worker_seeds='UNRECORDED', runtime_source='UNATTESTED', resume_digest='UNRECORDED')


def audit(root, plan):
    result = dict(status='FAIL', training='PAUSED', GPU_work=0, sessions=[], differences=[], failures=[], input_hashes={})
    previous = None
    for stage in STAGES:
        try:
            path = root / f'baseline-{stage}-20261008-seed20261009.zip'
            sha = digest(path.read_bytes())
            result['input_hashes'][path.name] = sha
            require(sha == plan['input_hashes'][path.name], 'Frozen input hash mismatch: ' + stage)
            previous = read_stage(path, stage, previous)
            result['sessions'].append(previous)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, zipfile.BadZipFile) as exc:
            result['failures'].append(f'{stage}: {exc}')
            break
    verdict = evaluate(result['sessions'], plan)
    result['differences'] = verdict['differences']
    result['failures'].extend(verdict['failures'])
    if not result['failures']:
        result['status'] = 'PASS'
    return result


def evaluate(sessions, plan):
    """Judge reader-verified evidence; synthetic evidence is used only in tests."""
    result = dict(status='FAIL', differences=[], failures=[])
    try:
        result['differences'] = compare(sessions, plan)
    except (ValueError, KeyError, TypeError) as exc:
        result['failures'].append('A/B comparison: ' + str(exc))
    for session in sessions:
        checks = [('worker_seeds', 'RECORDED', 'worker seeds were not independently recorded'),
                  ('runtime_source', 'ATTESTED', 'runtime source identity is not attested by the prepared source ZIP')]
        if session['initialization'] != 'fresh':
            checks.append(('resume_digest', 'RECORDED', 'resume input SHA256 is not recorded at load time'))
        for field, expected, gap in checks:
            if session.get(field) != expected:
                result['failures'].append(session['session_id'] + ': ' + gap)
    if not sessions:
        result['failures'].append('No verified sessions')
    if not result['failures']:
        result['status'] = 'PASS'
    return result


def execution_sessions(rows, meta, snapshot, plan):
    """Verify new startup evidence; never infer missing legacy evidence."""
    import io
    history = boundaries(rows)
    require(len(history) == len(plan['sessions']), 'Session count/boundary mismatch')
    source_sha = digest(snapshot)
    with zipfile.ZipFile(io.BytesIO(snapshot)) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'Duplicate source members')
        hashes = {n: digest(z.read(n)) for n in names if n.endswith('.py') and
                  (len(Path(n).parts) == 1 or (n.startswith('dougpu/') and len(Path(n).parts) == 2))}
    require('dougpu/train.py' in hashes, 'Missing trainer source')
    for session, approved in zip(history, plan['sessions']):
        own = [r for r in rows if r['session_id'] == session['session_id']]
        starts = [r for r in own if r['event'] == 'start']
        actors = [r for r in own if r['event'] == 'actor_start']
        ends = [r for r in own if r['event'] == 'session_end']
        require(len(starts) == len(actors) == len(ends) == 1, 'Missing/duplicate session evidence')
        require(own[-1] == ends[0] and ends[0]['stop_reason'] == 'target_updates'
                and ends[0]['update_endpoint'] == {
                    'target_updates': session['end_updates'], 'status': 'COMPLETE',
                    'stop_reason': 'target_updates'}, 'Incomplete session endpoint')
        start, actor = starts[0], actors[0]
        require(0 <= session['start_updates'] < session['end_updates'], 'Empty or invalid session')
        for key in ('start_updates', 'end_updates'):
            require(session[key] == approved[key], 'Unapproved restart boundary: ' + key)
        require(start['updates'] == actor['updates'] == session['start_updates'], 'Wrong actor boundary')
        require(own.index(actor) < own.index(start) and
                all(own.index(start) < i for i, row in enumerate(own) if row['event'] == 'train'),
                'Actor evidence recorded after training')
        require(start['train']['target_updates'] == session['end_updates'], 'Wrong target endpoint')
        require({'python', 'jax', 'numpy', 'backend'} <= set(approved['versions']),
                'Incomplete frozen runtime versions')
        for key, value in approved['versions'].items():
            require(start['versions'][key] == value, 'Unapproved runtime version: ' + key)
        for key in ('model', 'train', 'source_lock'):
            require(start[key] == approved[key], 'Unapproved execution config: ' + key)
        tc = TrainConfig(**start['train']).validate()
        require(set(start['train']) == {f.name for f in fields(TrainConfig)}, 'Incomplete train config')
        require(set(start['model']) == {f.name for f in fields(ModelConfig)}, 'Incomplete model config')
        ModelConfig(**start['model']).validate()
        require(actor['worker_order'] == list(range(tc.workers)), 'Worker order mismatch')
        require(len(actor['worker_seeds']) == tc.workers and
                all(type(seed) is int and 0 <= seed < 2**32-1 for seed in actor['worker_seeds']),
                'Missing/invalid worker seeds')
        require(actor['mode'] == ('ready_first' if tc.ready_first else 'ordered') and
                actor['packed'] == (not (tc.ready_first or tc.selfplay_kv_cache)), 'Actor mode mismatch')
        source = start['runtime_source']
        require(source['snapshot_sha256'] == approved['source_sha256'] == source_sha,
                'Unapproved source snapshot')
        require({n: v['sha256'] for n, v in source['files'].items()} == hashes, 'Runtime source mismatch')
        require(bool(source['loaded_modules']) and 'dougpu/train.py' in source['loaded_modules'].values(),
                'Missing loaded trainer path')
        for name in source['loaded_modules'].values():
            require(name in hashes and Path(source['files'][name]['path']).is_absolute(), 'Invalid loaded source path')
        resume = start['resume_input']
        if approved['input_checkpoint_sha256'] is None:
            require(resume is None and session['start_updates'] == 0, 'Unexpected resume input')
        else:
            require(resume is not None and resume['sha256'] == approved['input_checkpoint_sha256']
                    and resume['updates'] == session['start_updates'] and bool(resume['path']),
                    'Unapproved resume input SHA256/boundary')
    last = history[-1]
    require(meta['session_id'] == last['session_id'] and meta['updates'] == last['end_updates'],
            'Checkpoint/session mismatch')
    for key in ('runtime_source', 'resume_input', 'model', 'train', 'source_lock', 'versions'):
        require(meta[key] == start[key], 'Checkpoint/start disagreement: ' + key)
    require(isinstance(meta['actor_start'], dict), 'Missing checkpoint actor evidence')
    require(all(actor[k] == v for k, v in meta['actor_start'].items()) and
            set(meta['actor_start']) == {'worker_seeds', 'worker_order', 'mode', 'packed', 'updates', 'session_id'},
            'Checkpoint/actor disagreement')
    require(rows[-1]['event'] == 'session_end' and rows[-1]['stop_reason'] == 'target_updates'
            and meta['reason'] == 'session_end', 'Incomplete final session')
    require(meta['update_endpoint'] == rows[-1]['update_endpoint'] == {
        'target_updates': last['end_updates'], 'status': 'COMPLETE', 'stop_reason': 'target_updates'},
        'Incomplete endpoint')
    return history


def audit_execution(root, plan):
    from dougpu.checkpoint import Store
    from dougpu.semantics import check_array_state
    from dougpu.replay import Replay
    import numpy as np

    snapshot = (root/'source/trainer_source.zip').read_bytes()
    previous = None
    verified = []
    for index, approved in enumerate(plan['sessions']):
        sha = approved['checkpoint_sha256']
        require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), 'Invalid endpoint SHA256')
        directory = root/'endpoints'/sha
        require(directory.is_dir(), 'Missing independent endpoint directory')
        saved = Store(directory).load_latest()
        require(saved is not None and saved['sha256'] == sha, 'Wrong endpoint checkpoint / fallback')
        meta = saved['meta']
        require(meta['session_id'] == approved['session_id'], 'Wrong endpoint session_id')
        check_array_state(saved, ModelConfig(**meta['model']))
        replay = Replay(meta['train']['replay_capacity'])
        replay.restore(saved['replay'])
        rng = np.random.default_rng()
        rng.bit_generator.state = meta['numpy_rng']
        require(approved['input_checkpoint_sha256'] == (previous['sha256'] if previous else None),
                'Broken endpoint SHA256 chain')
        if previous:
            require(saved['log'].startswith(previous['log']), 'Restored endpoint history mismatch')
            require(meta['resume_input']['cycle'] == previous['meta']['cycle'], 'Wrong resume cycle')
            rng.bit_generator.state = previous['meta']['numpy_rng']
        else:
            rng = np.random.default_rng(meta['train']['seed'])
        expected_seeds = rng.integers(0, 2**32-1, meta['train']['workers'], dtype=np.uint64).tolist()
        require(isinstance(meta['actor_start'], dict) and
                meta['actor_start']['worker_seeds'] == expected_seeds, 'Actor seeds disagree with input RNG')
        rows = [json.loads(line) for line in saved['log'].splitlines()]
        history = execution_sessions(rows, meta, snapshot, dict(sessions=plan['sessions'][:index+1]))
        verified.append(dict(history[-1], checkpoint_sha256=sha))
        previous = saved
    require(previous is not None and previous['sha256'] == plan['checkpoint_sha256'], 'Wrong final endpoint')
    raw = (root/'metrics.jsonl').read_bytes()
    require(raw.startswith(previous['log']), 'Checkpoint/external log disagreement')
    # The operational save receipt follows termination; it is never completion evidence.
    tail = [json.loads(line) for line in raw[len(previous['log']):].splitlines()]
    require(len(tail) <= 1 and all(r['event'] == 'checkpoint' and r['reason'] == 'session_end'
            and r['session_id'] == previous['meta']['session_id']
            and r['updates'] == previous['meta']['updates'] for r in tail), 'Unexpected external log tail')
    return dict(status='PASS', training='PAUSED', GPU_work=0, sessions=verified,
                checkpoint_sha256=previous['sha256'], scope='execution evidence only; no experiment authorization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', action='store_true', help='Verify a new run against an approved evidence plan')
    parser.add_argument('baseline', type=Path)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    try:
        raw = args.plan.read_bytes()
        with redirect_stdout(sys.stderr):
            result = (audit_execution if args.execution else audit)(args.baseline, json.loads(raw))
        result['plan_sha256'] = digest(raw)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, zipfile.BadZipFile) as exc:
        result = {'status': 'FAIL', 'training': 'PAUSED', 'failures': [str(exc)], 'GPU_work': 0}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
