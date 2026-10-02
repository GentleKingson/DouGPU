"""Run the existing trainer unchanged in an isolated Colab/Linux environment.

From the repository root, use the environment's Python:
  python scripts/colab_benchmark.py --config input.config.json \
      --source-state input-state --run-dir benchmark --seconds 720
Only the session time budget changes. Never point --run-dir at a production run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_local import load_config
from dougpu.files import sha256_file
from dougpu.metrics_summary import summarize_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=720)
    args = parser.parse_args()
    if not 0 < args.seconds < float('inf'):
        parser.error('--seconds must be positive and finite')
    config = args.config.resolve()
    source = args.source_state.resolve()
    run = args.run_dir.resolve()
    spec, _, tc = load_config(config)
    if run.exists() or not source.is_dir() or not list(source.glob('*.ok.json')):
        parser.error('Use a new run directory and a complete, verified source checkpoint directory')
    if tc.backend != 'cuda':
        parser.error('This comparison requires the unchanged CUDA training configuration')
    run.mkdir(parents=True)
    original = config.read_bytes()
    source_hashes = {p.name: sha256_file(p)
                     for p in source.iterdir() if p.is_file()}

    def stage(name, *arguments):
        command = [sys.executable, str(ROOT / 'run_local.py'), *map(str, arguments)]
        print(name, ' '.join(command), flush=True)
        began = time.monotonic()
        with (run / (name + '.log')).open('w') as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - began
        if result.returncode:
            print((run / (name + '.log')).read_text()[-12000:], flush=True)
            result.check_returncode()
        return elapsed

    stage('prepare', 'prepare', '--config', config, '--run-dir', run)
    stage('import', 'import-checkpoint', '--source-state', source, '--run-dir', run)
    stage('doctor', 'doctor', '--config', config, '--output', run / 'hardware.json')
    stage('preflight', 'preflight', '--run-dir', run)
    wall = stage('train', 'train', '--run-dir', run, '--hours', args.seconds / 3600)
    session = json.loads((run / 'session_config.json').read_text())
    expected = json.loads(json.dumps(spec))
    expected['train']['max_hours'] = args.seconds / 3600
    assert session == expected, 'A setting other than the session time budget changed'
    assert config.read_bytes() == original, 'Input configuration changed'
    for name, digest in source_hashes.items():
        assert sha256_file(source / name) == digest
    summary = summarize_metrics(run / 'metrics.jsonl', config=session,
                                process_returncode=0, process_wall_seconds=wall,
                                require_evaluation=tc.eval_every > 0)
    summary['configuration_unchanged_except_session_budget'] = True
    summary['input_config_sha256'] = hashlib.sha256(original).hexdigest()
    summary['source_checkpoint_hashes'] = source_hashes
    summary['measurement'] = 'Single short trial; full cycles after 25 warmup cycles, not GPU-only speed.'
    (run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    if not summary['eligible']:
        raise RuntimeError('Measurement is not eligible; inspect summary.json')


if __name__ == '__main__':
    main()
