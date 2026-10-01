"""The GPU migration preflight must stress valid tokens, not just PAD widths."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dougpu.config import ModelConfig, TrainConfig
from dougpu.preflight import HISTORY_BUCKETS, synthetic_training_batch


@pytest.mark.parametrize('groups', [1, 4])
def test_preflight_full_512_is_real_non_pad_work(groups):
    config = TrainConfig(backend='cpu', micro_batch=64, accumulation=4, history_groups=groups)
    for length in HISTORY_BUCKETS:
        generated = synthetic_training_batch(config, length)
        batches = generated if isinstance(generated, tuple) else (generated,)
        assert sum(batch['tokens'].shape[0] * batch['tokens'].shape[1] for batch in batches) == 256
        for batch in batches:
            assert batch['tokens'].shape == (4 // groups, 64, length)
            assert batch['tokens'].dtype == batch['lengths'].dtype == np.dtype('int32')
            np.testing.assert_array_equal(np.count_nonzero(batch['tokens'], axis=-1), batch['lengths'])
            assert np.all(batch['lengths'] == length)
            assert np.all(batch['tokens'] > 0) and np.all(batch['tokens'] < 23)
        roles = np.concatenate([batch['role'].reshape(-1) for batch in batches])
        weights = np.concatenate([batch['weight'].reshape(-1) for batch in batches])
        for role in range(3):
            np.testing.assert_allclose(weights[roles == role].sum(), 256 / 3, rtol=1e-6)


def _command(tmp_path, spec, *, repeats=1):
    config = tmp_path / 'config.json'
    report = tmp_path / 'report.json'
    config.write_text(json.dumps(spec))
    command = [sys.executable, '-m', 'dougpu.preflight', '--config', str(config),
               '--cache-dir', str(tmp_path / 'cache'), '--output', str(report),
               '--warm-repeats', str(repeats)]
    env = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    return command, env, report


def test_cpu_preflight_subprocess_verifies_all_real_lengths(tmp_path):
    mc = ModelConfig(width=8, layers=1, heads=2, ffn=16, q_hidden=8, bf16=False, remat=False)
    tc = TrainConfig(engine='reference', backend='cpu', attention_impl='xla',
                     workers=1, envs_per_worker=1, micro_batch=3, accumulation=1,
                     infer_batch=2, action_chunk=8)
    command, env, report = _command(tmp_path, {'model': asdict(mc), 'train': asdict(tc)})
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                               text=True, capture_output=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    doc = json.loads(report.read_text())
    assert doc['status'] == 'pass' and doc['backend'] == 'cpu'
    assert doc['gpu_validation_executed'] is False
    assert doc['attention_parity']['status'] == 'pass'
    assert doc['attention_parity']['all_gradients_finite_fp32'] is True
    assert doc['verified_optimizer_updates'] == 8
    assert [row['token_width'] for row in doc['training_shapes']] == list(HISTORY_BUCKETS)
    for row in doc['training_shapes']:
        assert row['actual_valid_lengths'] == [row['token_width']]
        assert row['all_tokens_non_pad'] is True
        assert row['minimum_max_parameter_abs_change'] > 0
        assert row['compile_plus_first_step_seconds'] > 0 and row['warm_mean_step_seconds'] > 0
    assert all(row['best_tail_index'] == 8 and row['tie_first_index'] == 0
               for row in doc['inference_shapes'])


def test_preflight_failure_writes_error_report_and_returns_nonzero(tmp_path):
    command, env, report = _command(tmp_path, {'train': {'backend': 'unavailable'}})
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                               text=True, capture_output=True, timeout=30)
    assert completed.returncode != 0
    doc = json.loads(report.read_text())
    assert doc['status'] == 'error' and doc['stage'] == 'configuration'
    assert doc['error']['type'] == 'ValueError' and 'backend' in doc['error']['message']
