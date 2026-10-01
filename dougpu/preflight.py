"""Validate the selected device and time production-shaped synthetic updates.

Every training token in the 64/128/256/512 stress batches is non-PAD, including
the full 512-token case. This is a capacity/kernel check, not game data or a
self-play speed measurement. No preflight learning state is saved for training.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
from .config import ModelConfig, TrainConfig
from .files import write_json
from .runtime import (configure_runtime, nvidia_info, package_versions,
                      requested_backend, verify_backend)

HISTORY_BUCKETS = (64, 128, 256, 512)


def synthetic_training_batch(config, length, *, vocab=32, seed=1907):
    """Full non-PAD histories with production accumulation and replay role weights."""
    import numpy as np
    from .encoding import ACTION_DIM, BELIEF_DIM, STATE_DIM

    if length not in HISTORY_BUCKETS:
        raise ValueError('Preflight length must be one of 64, 128, 256, 512')
    config.validate()
    if vocab < 23:
        raise ValueError('Preflight requires the complete game token vocabulary')
    rng = np.random.default_rng(seed + length)
    shape = (config.accumulation, config.micro_batch)
    roles = (np.arange(config.batch_size, dtype=np.int32) % 3).reshape(shape)
    counts = np.bincount(roles.reshape(-1), minlength=3)
    tokens = rng.integers(1, 23, shape + (length,), dtype=np.int32)
    tokens[..., 0] = 1
    batch = {
        'tokens': tokens, 'lengths': np.full(shape, length, np.int32),
        'state': rng.random(shape + (STATE_DIM,), dtype=np.float32),
        'actions': rng.random(shape + (ACTION_DIM,), dtype=np.float32),
        'role': roles, 'belief': rng.random(shape + (BELIEF_DIM,), dtype=np.float32),
        'target': np.where(roles == 0, 1., -1.).astype(np.float32),
        'weight': (config.batch_size / (3 * counts[roles])).astype(np.float32),
    }
    if config.history_groups == 1:
        return batch
    split = {key: np.split(value, config.history_groups, axis=0) for key, value in batch.items()}
    return tuple({key: pieces[index] for key, pieces in split.items()}
                 for index in range(config.history_groups))


def _relative_l2(actual, expected):
    import numpy as np

    actual = np.asarray(actual, np.float64).reshape(-1)
    expected = np.asarray(expected, np.float64).reshape(-1)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FloatingPointError('Non-finite values in attention parity probe')
    return float(np.linalg.norm(actual - expected) /
                 max(float(np.linalg.norm(expected)), 1e-7))


def attention_parity(model_config, train_config, params):
    """Compare the configured model's outputs, losses and all gradients on-device.

    Use a small ragged batch before timing. Full production batch and 512-token
    capacity are exercised separately. BF16 tolerances do not claim bit equality.
    """
    import jax
    import numpy as np
    from .encoding import ACTION_DIM, BELIEF_DIM, STATE_DIM
    from .inference import validate_attention_tokens
    from .model import encode_history, sample_losses

    implementation = train_config.attention_impl
    if implementation == 'manual':
        return {'status': 'not_required', 'implementation': 'manual',
                'reason': 'The unchanged notebook attention is the reference implementation.'}
    rng = np.random.default_rng(train_config.seed + 971)
    lengths = np.asarray([6, 33, 64], np.int32)
    tokens = rng.integers(1, 23, (3, 64), dtype=np.int32)
    tokens[np.arange(64)[None, :] >= lengths[:, None]] = 0
    tokens[:, 0] = 1
    batch = {'tokens': tokens, 'lengths': lengths,
             'state': rng.random((3, STATE_DIM), dtype=np.float32),
             'actions': rng.random((3, ACTION_DIM), dtype=np.float32),
             'role': np.arange(3, dtype=np.int32),
             'belief': rng.random((3, BELIEF_DIM), dtype=np.float32),
             'target': np.asarray([1., -1., -1.], np.float32),
             'weight': np.asarray([.8, 1., 1.2], np.float32)}
    validate_attention_tokens(tokens, lengths)
    mc = replace(model_config, remat=False)
    outputs, gradients, histories, functions = {}, {}, {}, {}
    started = time.perf_counter()
    for name in ('manual', implementation):
        config = replace(train_config, attention_impl=name)
        history_fn = jax.jit(lambda p, t, l, impl=name: encode_history(
            p, t, l, mc, attention_impl=impl))
        gradient_fn = jax.jit(jax.value_and_grad(
            lambda p, b, cfg=config: sample_losses(p, b, mc, cfg), has_aux=True))
        histories[name] = jax.device_get(history_fn(params, tokens, lengths))
        outputs[name], gradients[name] = jax.device_get(gradient_fn(params, batch))
        functions[name] = history_fn

    valid = tokens != 0
    context_error = _relative_l2(histories[implementation][0], histories['manual'][0])
    hidden_error = _relative_l2(histories[implementation][1][valid], histories['manual'][1][valid])
    activation_limit, gradient_limit = ((.04, .08) if mc.bf16 else (1e-4, 3e-4))
    if max(context_error, hidden_error) > activation_limit:
        raise AssertionError(f'{implementation} activation parity failed: context={context_error:g}, '
                             f'valid_hidden={hidden_error:g}, limit={activation_limit:g}')
    for name, (loss, metrics) in outputs.items():
        if (np.asarray(loss).dtype != np.dtype('float32') or metrics.dtype != np.dtype('float32')
                or not np.isfinite(loss).all() or not np.isfinite(metrics).all()):
            raise FloatingPointError('Non-finite or non-FP32 attention loss: ' + name)
    reference_metrics, candidate_metrics = outputs['manual'][1], outputs[implementation][1]
    np.testing.assert_allclose(candidate_metrics, reference_metrics,
                               rtol=.03 if mc.bf16 else 3e-5,
                               atol=5e-4 if mc.bf16 else 2e-6,
                               err_msg='Attention loss parity failed')
    reference, candidate = gradients['manual'], gradients[implementation]
    if set(reference) != set(candidate) or set(candidate) != set(params):
        raise AssertionError('Attention gradient parameter set changed')
    for name in sorted(params):
        for values in (reference[name], candidate[name]):
            if values.dtype != np.dtype('float32') or not np.isfinite(values).all():
                raise FloatingPointError('Non-finite or non-FP32 attention gradient: ' + name)
    reference_vector = np.concatenate([reference[name].reshape(-1) for name in sorted(params)])
    candidate_vector = np.concatenate([candidate[name].reshape(-1) for name in sorted(params)])
    gradient_error = _relative_l2(candidate_vector, reference_vector)
    if gradient_error > gradient_limit:
        raise AssertionError(f'{implementation} gradient parity failed: {gradient_error:g} > {gradient_limit:g}')
    changed = tokens.copy()
    changed[2, 24:] = 1 + changed[2, 24:] % 22
    _, changed_hidden = jax.device_get(functions[implementation](params, changed, lengths))
    causal_error = _relative_l2(changed_hidden[2, :24], histories[implementation][1][2, :24])
    if causal_error > 1e-6:
        raise AssertionError(f'Future tokens changed the common attention prefix: {causal_error:g}')
    return {'status': 'pass', 'implementation': implementation, 'reference': 'manual',
            'device_backend': jax.default_backend(), 'batch': 3, 'token_width': 64,
            'valid_lengths': lengths.tolist(), 'model': asdict(mc),
            'context_relative_l2': context_error, 'valid_hidden_relative_l2': hidden_error,
            'activation_relative_l2_limit': activation_limit,
            'all_gradient_relative_l2': gradient_error, 'all_gradient_relative_l2_limit': gradient_limit,
            'max_gradient_abs_difference': float(np.max(np.abs(candidate_vector - reference_vector))),
            'gradient_tensor_count': len(candidate), 'all_gradients_finite_fp32': True,
            'gradient_comparison': 'Vector containing every trainable parameter gradient',
            'reference_losses': np.asarray(reference_metrics, np.float32).tolist(),
            'candidate_losses': np.asarray(candidate_metrics, np.float32).tolist(),
            'causal_prefix_relative_l2': causal_error,
            'compile_and_check_seconds': time.perf_counter() - started,
            'note': 'Small-batch numerical gate on the reported device; not bitwise or strength equivalence.'}


def _check_update(previous_params, previous_step, params, optimizer, metrics):
    """Validate all update outputs outside the measured kernel interval."""
    import jax
    import numpy as np

    before, after, opt, values = jax.device_get((previous_params, params, optimizer, metrics))
    values = np.asarray(values)
    if values.shape != (6,) or not np.isfinite(values).all() or values[-1] != 1:
        raise FloatingPointError('Preflight optimizer update was skipped or non-finite')
    if int(opt['step']) != previous_step + 1:
        raise AssertionError('Preflight Adam step did not advance exactly once')
    for label, arrays in (('parameters', after), ('Adam m', opt['m']), ('Adam v', opt['v'])):
        if not all(np.isfinite(np.asarray(value)).all() for value in arrays.values()):
            raise FloatingPointError('Non-finite ' + label + ' after preflight update')
    change = max(float(np.max(np.abs(np.asarray(after[key]) - np.asarray(before[key])))) for key in before)
    if change == 0:
        raise AssertionError('Preflight optimizer step left every parameter unchanged')
    return {'optimizer_step': int(opt['step']), 'max_parameter_abs_change': change,
            'losses': values[:4].tolist(), 'grad_norm': float(values[4])}


def run_preflight(model_config, train_config, cache_dir, *, warm_repeats=3, report=None):
    if warm_repeats < 1:
        raise ValueError('--warm-repeats must be positive')
    report = {} if report is None else report
    report['stage'] = 'backend_initialization'
    configure_runtime(train_config, cache_dir)
    import jax
    import numpy as np
    from .encoding import ACTION_DIM, STATE_DIM
    from .inference import Inference, validate_attention_tokens
    from .model import init_optimizer, init_params, make_train_step

    devices = verify_backend(train_config, jax)
    report.update({'status': 'running', 'requested_backend': requested_backend(train_config),
                   'backend': jax.default_backend(), 'jax_version': jax.__version__,
                   'devices': [str(device) for device in devices],
                   'device_kind': devices[0].device_kind, 'packages': package_versions(),
                   'nvidia_smi': nvidia_info(), 'gpu_validation_executed': jax.default_backend() == 'gpu',
                   'attention_impl': train_config.attention_impl, 'model': asdict(model_config),
                   'effective_batch': train_config.batch_size,
                   'micro_batch': train_config.micro_batch, 'accumulation': train_config.accumulation,
                   'history_groups': train_config.history_groups,
                   'learner_remat': train_config.learner_remat,
                   'effective_learner_remat': (model_config.remat if train_config.learner_remat is None
                                             else train_config.learner_remat),
                   'timing_scope': 'Device dispatch through block_until_ready; host output validation excluded',
                   'note': 'Synthetic kernel/capacity benchmark, not end-to-end self-play throughput or playing strength.'})
    params = init_params(model_config, train_config.seed)
    report['parameters'] = sum(value.size for value in params.values())
    report['stage'] = 'attention_parity'
    print(f'[PREFLIGHT] {report["device_kind"]}; attention={train_config.attention_impl}; numerical gate', flush=True)
    report['attention_parity'] = attention_parity(model_config, train_config, params)

    report['stage'] = 'production_inference'
    infer = Inference(model_config, train_config.infer_batch, train_config.action_chunk,
                      history_buckets=train_config.infer_history_buckets,
                      fused=train_config.infer_fused, attention_impl=train_config.attention_impl)
    # Zero output projections create exact ties, independent of GEMM row rounding.
    tied_params = dict(params, **{'qout.w': np.zeros_like(params['qout.w']),
                                'qout.b': np.zeros_like(params['qout.b'])})
    before = time.perf_counter()
    report['inference_shapes'] = inference_shapes = []
    for length in HISTORY_BUCKETS:
        print(f'[PREFLIGHT] production inference length={length}', flush=True)
        req = {'tokens': np.resize(np.arange(1, 23, dtype=np.int32), length),
               'state': np.zeros(STATE_DIM, np.float32), 'role': 0,
               'actions': np.zeros((1, ACTION_DIM), np.float32)}
        sizes = sorted({train_config.infer_batch} |
                       ({min(size, train_config.infer_batch) for size in (32, 64)}
                        if train_config.infer_history_buckets else set()))
        for size in sizes:
            if infer.choose(params, [req] * size) != [0] * size:
                raise AssertionError(f'Invalid inference choice at length={length}, batch={size}')
        # The best candidate is AFTER the first full chunk. Truncation cannot
        # pass this check, even if the omitted actions happened to be tied.
        probe = dict(req, actions=np.stack([np.full(ACTION_DIM, value, np.float32)
                                           for value in (0., .25, .5, 1.)]))
        scores = infer.score(params, [probe])[0]
        low, high = int(np.argmin(scores)), int(np.argmax(scores))
        if not scores[high] > scores[low]:
            raise AssertionError('Synthetic overflow probe produced no distinguishable action scores')
        overflow = dict(req, actions=np.repeat(probe['actions'][low:low + 1],
                                              train_config.action_chunk + 1, axis=0))
        overflow['actions'][-1] = probe['actions'][high]
        if infer.choose(params, [overflow]) != [train_config.action_chunk]:
            raise AssertionError('Preflight failed to select the best action beyond the first chunk')
        tied = dict(req, actions=np.repeat(probe['actions'][low:low + 1],
                                          train_config.action_chunk + 1, axis=0))
        if infer.choose(tied_params, [tied]) != [0]:
            raise AssertionError('Preflight cross-chunk first-index tie-break failed')
        inference_shapes.append({'valid_history_length': length, 'batches': sizes,
                                 'overflow_action_count': train_config.action_chunk + 1,
                                 'best_tail_index': train_config.action_chunk, 'tie_first_index': 0,
                                 'status': 'pass'})
    report['inference_bucket_compile_and_checks_seconds'] = time.perf_counter() - before

    report['stage'] = 'production_training'
    optimizer, step = init_optimizer(params), make_train_step(model_config, train_config)
    report['training_shapes'] = training_shapes = []
    optimizer_steps = 0
    for length in HISTORY_BUCKETS:
        report['active_training_valid_length'] = length
        print(f'[PREFLIGHT] learner length={length}, micro={train_config.micro_batch}, '
              f'accumulation={train_config.accumulation}, groups={train_config.history_groups}', flush=True)
        host_batch = synthetic_training_batch(train_config, length,
                                              vocab=model_config.vocab, seed=train_config.seed)
        groups = host_batch if isinstance(host_batch, tuple) else (host_batch,)
        for group in groups:
            validate_attention_tokens(group['tokens'], group['lengths'])
            if not np.all(group['lengths'] == length) or not np.all(group['tokens'] != 0):
                raise AssertionError('Preflight must use actual non-PAD tokens throughout every history')
        batch = jax.device_put(host_batch)
        jax.block_until_ready(batch)
        previous, previous_step = params, optimizer_steps
        before = time.perf_counter()
        params, optimizer, metrics = jax.block_until_ready(step(params, optimizer, batch))
        compile_first_seconds = time.perf_counter() - before
        first_check = _check_update(previous, previous_step, params, optimizer, metrics)
        optimizer_steps = first_check['optimizer_step']
        times, minimum_change = [], first_check['max_parameter_abs_change']
        for _ in range(warm_repeats):
            previous, previous_step = params, optimizer_steps
            before = time.perf_counter()
            params, optimizer, metrics = jax.block_until_ready(step(params, optimizer, batch))
            times.append(time.perf_counter() - before)
            checked = _check_update(previous, previous_step, params, optimizer, metrics)
            optimizer_steps = checked['optimizer_step']
            minimum_change = min(minimum_change, checked['max_parameter_abs_change'])
        mean_seconds = float(np.mean(times))
        training_shapes.append({'status': 'pass', 'token_width': length,
                                'actual_valid_lengths': [length], 'all_tokens_non_pad': True,
                                'group_tensor_shapes': [list(group['tokens'].shape) for group in groups],
                                'effective_batch': train_config.batch_size,
                                'compile_plus_first_step_seconds': compile_first_seconds,
                                'warm_step_seconds': times, 'warm_mean_step_seconds': mean_seconds,
                                'warm_median_step_seconds': float(np.median(times)),
                                'synthetic_samples_per_second': train_config.batch_size / mean_seconds,
                                'verified_optimizer_updates': warm_repeats + 1,
                                'minimum_max_parameter_abs_change': minimum_change,
                                'first_update': first_check, 'last_update': checked})
    report.pop('active_training_valid_length', None)
    report['verified_optimizer_updates'] = optimizer_steps
    report['group_shape_coverage'] = {
        'warmed_signatures': [[length] * train_config.history_groups for length in HISTORY_BUCKETS],
        'possible_sorted_bucket_signatures_upper_bound': math.comb(train_config.history_groups + 3, 3),
        'mixed_group_signatures_exhaustively_warmed': train_config.history_groups == 1,
        'note': ('Only all-equal-width tuples are compiled here; mixed history groups can still compile during training.'
                 if train_config.history_groups > 1 else 'All four ungrouped history bucket widths are exercised.')}
    report['jax_memory_stats'] = devices[0].memory_stats()
    report['memory_stats_scope'] = 'End of entire preflight; includes parity probes and compiled kernel caches.'
    report['stage'] = 'complete'
    report['status'] = 'pass'
    return report


def _write_report(path, report):
    if path:
        write_json(path, report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--output', default='')
    ap.add_argument('--warm-repeats', type=int, default=3,
                    help='Synchronized, validated warm updates per production history bucket')
    args = ap.parse_args()
    started = time.perf_counter()
    report = {'status': 'running', 'stage': 'configuration', 'config_path': str(Path(args.config).resolve())}
    try:
        spec = json.loads(Path(args.config).read_text())
        mc = ModelConfig(**spec.get('model', {})).validate()
        tc = TrainConfig(**spec.get('train', {})).validate()
        report['requested_backend'] = requested_backend(tc)
        run_preflight(mc, tc, args.cache_dir, warm_repeats=args.warm_repeats, report=report)
        report['total_preflight_seconds'] = time.perf_counter() - started
        _write_report(args.output, report)
    except BaseException as exc:
        report.update({'status': 'error', 'error': {'type': type(exc).__name__, 'message': str(exc)},
                       'total_preflight_seconds': time.perf_counter() - started})
        try:
            _write_report(args.output, report)
        except Exception as write_error:
            report['report_write_error'] = str(write_error)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
        raise
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
