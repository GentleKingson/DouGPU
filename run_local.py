#!/usr/bin/env python3
"""DouGPU local single-GPU workflow. Run `python run_local.py --help` for the stages."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
from dougpu.runtime import thread_environment, configure_runtime, diagnostics
thread_environment()


def add_vendor_path():
    vendor = str(ROOT/'vendor')
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    existing = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, (str(ROOT), vendor, existing)))


def load_config(path):
    from dougpu.config import ModelConfig, TrainConfig
    spec = json.loads(Path(path).read_text())
    if set(spec) != {'model', 'train'}:
        raise ValueError('Configuration must contain model and train objects only')
    mc = ModelConfig(**spec['model']).validate()
    tc = TrainConfig(**spec['train']).validate()
    if tc.attention_impl == 'cudnn' and not mc.bf16:
        raise ValueError('The cuDNN candidate requires BF16. Keep manual attention for FP32 checks.')
    return {'model': asdict(mc), 'train': asdict(tc)}, mc, tc


@contextmanager
def run_lock(run):
    """Avoid two learners, or a prepare/import operation, writing the same run."""
    import fcntl
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    with open(run/'.run.lock', 'a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This run is already in use by another local process: '+str(run)) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def gpu_lock(tc):
    """Serialize this project's CUDA owners, including independent experiments."""
    import fcntl
    from dougpu.runtime import requested_backend
    if requested_backend(tc) != 'cuda':
        yield
        return
    root = Path(os.environ.get('DOUGPU_LOCK_DIR', str(ROOT/'.cache'/'locks')))
    root.mkdir(parents=True, exist_ok=True)
    # A conservative project lock avoids CUDA_VISIBLE_DEVICES aliases bypassing ownership.
    with open(root/'single-gpu.lock', 'a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another DouGPU CUDA process is running; finish it before this stage.') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def write_json(path, value):
    from dougpu.checkpoint import atomic_bytes, json_bytes
    atomic_bytes(path, json_bytes(value))


def archive_trainer(run):
    """Keep the source snapshot from prepare; later sessions record their own versions."""
    destination = run/'source'/'trainer_source.zip'
    if destination.exists():
        return
    files = list((ROOT/'dougpu').glob('*.py')) + list(ROOT.glob('*.py'))
    files += list((ROOT/'configs').glob('*.json')) + list(ROOT.glob('requirements*.txt'))
    files += list((ROOT/'scripts').glob('*'))
    files += [ROOT/name for name in ('LICENSE', 'constraints-cuda13.txt', 'Dockerfile', 'compose.yaml')
              if (ROOT/name).is_file()]
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for file in sorted(files):
            z.write(file, str(file.relative_to(ROOT)))


def prepared(run):
    if not (run/'config.json').is_file() or not (run/'source'/'source_lock.json').is_file():
        raise FileNotFoundError('Run is not prepared. Use prepare --run-dir ... --config ... first.')
    spec, mc, tc = load_config(run/'config.json')
    lock = json.loads((run/'source'/'source_lock.json').read_text())
    from bootstrap import verify_local_source
    if tc.engine == 'douzero':
        verify_local_source(ROOT, lock)
    elif lock.get('engine') != tc.engine or lock.get('encoding_schema') != 1:
        raise ValueError('Reference-test source lock changed')
    return spec, mc, tc, lock


def prepare_run(args):
    from bootstrap import prepare, PINNED_COMMIT
    run = Path(args.run_dir).resolve()
    spec, mc, tc = load_config(args.config)
    with run_lock(run):
        if (run/'config.json').exists() and load_config(run/'config.json')[0] != spec:
            raise ValueError('Run already has a different configuration. Use a new run directory.')
        if tc.engine == 'douzero':
            cache = args.upstream_cache or (ROOT/'upstream_cache' if (ROOT/'upstream_cache').is_dir() else None)
            prepare(ROOT, run/'source', PINNED_COMMIT, cache)
        elif (run/'source'/'source_lock.json').exists():
            prepared(run)
        else:
            write_json(run/'source'/'source_lock.json', {'engine': 'reference', 'encoding_schema': 1})
        write_json(run/'config.json', spec)
        archive_trainer(run)
        write_json(run/'run_info.json', {'project': 'DouGPU', 'run_dir': str(run),
                   'source_notebook': 'DouTPU_v6e1_Optimized_Final.ipynb',
                   'default_backend': tc.backend, 'fresh_start_unless_checkpoint_imported': True})
    print('PREPARED', run, flush=True)


def import_checkpoint(args):
    from dougpu.checkpoint import Store
    from dougpu.semantics import check_array_state
    from fork_run import fork
    run, source = Path(args.run_dir).resolve(), Path(args.source_state).resolve()
    if not source.is_dir():
        raise ValueError('--source-state must be a checkpoint directory containing ZIP + .ok.json, not a policy NPZ.')
    with run_lock(run):
        spec, mc, tc, lock = prepared(run)
        saved = Store(source).load_latest()
        if saved is None:
            raise FileNotFoundError('No full, verified checkpoint in source. A notebook or policy-only NPZ cannot restore training.')
        check_array_state(saved, mc)
        if saved['replay'] is None:
            raise ValueError('Full-state migration requires a checkpoint containing replay.')
        fork(source, run, None, spec, lock)


def invoke(module, argv):
    # A separate process has exclusive GPU ownership; actors spawned by train import
    # dougpu.train without importing/initializing JAX in its module body.
    command = [sys.executable, '-m', module, *map(str, argv)]
    # Terminal Ctrl+C must not interrupt the spawned CPU actors directly.
    # The launcher forwards to the learner PID, whose handler requests a safe stop.
    child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    previous = {}
    def forward(signum, _frame):
        if child.poll() is None:
            print('Stop requested; waiting for the learner to checkpoint at a safe point.', flush=True)
            child.send_signal(signum)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, forward)
    try:
        code = child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if code:
        raise subprocess.CalledProcessError(code, command)
    return subprocess.CompletedProcess(command, code)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Create an isolated run and verify the pinned rules source')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--upstream-cache', default='')
    p = commands.add_parser('import-checkpoint', help='Migrate complete TPU/local state into a NEW prepared run')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--source-state', required=True)
    p = commands.add_parser('doctor', help='Verify actual accelerator and execute a BF16 kernel')
    p.add_argument('--config', default=str(ROOT/'configs'/'rtx5070_balanced.json'))
    p.add_argument('--output', default='hardware.json')
    p = commands.add_parser('preflight', help='Check production inference and a truly full 512-token backward pass')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--config', default='')
    p = commands.add_parser('train', help='Train or resume locally; --cycles is ADDITIONAL cycles this session')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--config', default='')
    p.add_argument('--hours', type=float)
    p.add_argument('--cycles', type=int)
    p.add_argument('--mirror-dir', default='', help='Optional checkpoint mirror; local checkpoints always remain')
    p = commands.add_parser('selftest', help='Rules/encoding checks only; never starts JAX')
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--engine', choices=['reference', 'douzero'], default='douzero')
    p = commands.add_parser('evaluate', help='Evaluate on held-out paired deals (no training)')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--policy', choices=['latest', 'best'], default='best')
    p.add_argument('--opponent', choices=['rule', 'random', 'douzero'], default='rule')
    p.add_argument('--weights', default='')
    p.add_argument('--deals', type=int, default=1000)
    p.add_argument('--seed', type=int, default=910001)
    p.add_argument('--output', default='')
    args = ap.parse_args()
    add_vendor_path()
    if args.command == 'prepare':
        return prepare_run(args)
    if args.command == 'import-checkpoint':
        return import_checkpoint(args)
    if args.command == 'selftest':
        return invoke('dougpu.selftest', ['--games', args.games, '--engine', args.engine])
    if args.command == 'doctor':
        _, _, tc = load_config(args.config)
        try:
            configure_runtime(tc, ROOT/'.cache'/'jax')
            with gpu_lock(tc):
                report = dict(diagnostics(tc, ROOT/'.cache'/'jax'), status='passed')
        except Exception as exc:
            from dougpu.runtime import nvidia_info, package_versions, requested_backend
            write_json(args.output, {'status': 'failed', 'requested_backend': requested_backend(tc),
                                    'error_type': type(exc).__name__, 'error': str(exc),
                                    'nvidia_smi': nvidia_info(), 'packages': package_versions()})
            raise
        write_json(args.output, report)
        print(json.dumps(report, indent=2), flush=True)
        return
    run = Path(args.run_dir).resolve()
    with run_lock(run):
        spec, mc, tc, lock = prepared(run)
        if getattr(args, 'config', ''):
            candidate, cmc, ctc = load_config(args.config)
            from dougpu.semantics import check_training_semantics
            if asdict(cmc) != asdict(mc):
                raise ValueError('Model must remain identical for hardware tuning')
            check_training_semantics(spec['train'], ctc)
            if ctc.sample_credit != tc.sample_credit or ctc.replay_version_updates != tc.replay_version_updates:
                raise ValueError('Sampling/version semantics must remain identical for hardware tuning')
            spec, mc, tc = candidate, cmc, ctc
        configure_runtime(tc, ROOT/'.cache'/'jax')
        with gpu_lock(tc):
            if args.command == 'preflight':
                config = run/'preflight_config.json'
                write_json(config, spec)
                return invoke('dougpu.preflight', ['--config', config, '--cache-dir', os.environ['JAX_COMPILATION_CACHE_DIR'],
                                                    '--output', run/'preflight.json'])
            if args.command == 'evaluate':
                from dougpu.evaluation import validate_holdout_seed
                validate_holdout_seed(args.seed, [tc.eval_seed])
                output = args.output or str(run/f'evaluation_{args.opponent}_{args.seed}.json')
                return invoke('dougpu.evaluate', ['--policy', run/'checkpoints'/f'{args.policy}_policy.npz',
                    '--opponent', args.opponent, '--weights', args.weights, '--engine', tc.engine,
                    '--deals', args.deals, '--seed', args.seed, '--backend', tc.backend or ('tpu' if tc.require_tpu else 'cpu'),
                    '--attention-impl', tc.attention_impl, '--batch', tc.infer_batch,
                    '--action-chunk', tc.action_chunk, '--output', output])
            from dougpu.checkpoint import Store
            mirror = Path(args.mirror_dir).resolve() if args.mirror_dir else None
            if mirror and (mirror == (run/'checkpoints').resolve() or mirror == run):
                raise ValueError('Checkpoint mirror must be a different directory')
            # Match the learner's candidate set, including a newer valid mirror.
            saved = Store(run/'checkpoints', mirror).load_latest()
            if args.hours is not None:
                if args.hours <= 0:
                    ap.error('--hours must be positive')
                tc.max_hours = args.hours
            if args.cycles is not None:
                if args.cycles <= 0:
                    ap.error('--cycles must be positive')
                tc.max_cycles = (saved['meta']['cycle'] if saved else 0) + args.cycles
            tc.validate()
            spec['train'] = asdict(tc)
            config = run/'session_config.json'
            write_json(config, spec)
            argv = ['--config', config, '--workdir', run, '--source-lock', run/'source'/'source_lock.json']
            if mirror:
                argv += ['--savedir', mirror]
            return invoke('dougpu.train', argv)


if __name__ == '__main__':
    main()
