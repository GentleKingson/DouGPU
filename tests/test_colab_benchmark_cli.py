"""Run with: python tests/test_colab_benchmark_cli.py (stdlib only)."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
command = [sys.executable, str(ROOT / 'scripts/colab_benchmark.py'),
           '--config', str(ROOT / 'configs/rtx5070_throughput.json')]


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / 'source'
        source.mkdir()
        (source / 'checkpoint.ok.json').write_text('{}')
        run = root / 'run'
        args = command + ['--source-state', str(source), '--run-dir', str(run)]
        for seconds in ('0', '-1', 'nan', 'inf'):
            result = subprocess.run(args + ['--seconds', seconds], capture_output=True, text=True)
            assert result.returncode == 2 and 'positive and finite' in result.stderr
            assert not run.exists()
        run.mkdir()
        sentinel = run / 'keep.txt'
        sentinel.write_text('do not overwrite')
        result = subprocess.run(args, capture_output=True, text=True)
        assert result.returncode == 2 and 'new run directory' in result.stderr
        assert sentinel.read_text() == 'do not overwrite'
    print('Colab benchmark CLI guards passed')


if __name__ == '__main__':
    main()
