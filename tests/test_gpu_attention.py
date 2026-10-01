"""Backend migration parity; the cuDNN numerical gate requires a real CUDA GPU.

CPU tests establish the explicit XLA path against the unchanged notebook math.
They do not establish cuDNN support, GPU speed, or bitwise BF16 equivalence.
"""
from dataclasses import asdict

import jax
import numpy as np
import pytest

from dougpu.config import ModelConfig, TrainConfig
from dougpu.encoding import ACTION_DIM, BELIEF_DIM, STATE_DIM
from dougpu.inference import Inference, validate_attention_tokens
from dougpu.model import (encode_history, init_optimizer, init_params,
                          make_train_step, sample_losses)


@pytest.fixture(scope='module')
def small_policy():
    mc = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16,
                     bf16=False, remat=False)
    return mc, init_params(mc, 1207)


def histories(lengths, width, seed=13):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, np.int32)
    tokens = rng.integers(1, 23, (len(lengths), width), dtype=np.int32)
    tokens[np.arange(width)[None, :] >= lengths[:, None]] = 0
    tokens[:, 0] = 1
    return tokens, lengths


def loss_batch(lengths=(6, 33, 64), width=64):
    tokens, lengths = histories(lengths, width)
    rng = np.random.default_rng(773)
    n = len(lengths)
    return {'tokens': tokens, 'lengths': lengths,
            'state': rng.random((n, STATE_DIM), dtype=np.float32),
            'actions': rng.random((n, ACTION_DIM), dtype=np.float32),
            'role': np.arange(n, dtype=np.int32) % 3,
            'belief': rng.random((n, BELIEF_DIM), dtype=np.float32),
            'target': np.where(np.arange(n) % 3 == 0, 1., -1.).astype(np.float32),
            'weight': np.linspace(.8, 1.2, n, dtype=np.float32)}


def training_config(implementation, accumulation=1):
    tc = TrainConfig(micro_batch=3, accumulation=accumulation)
    tc.attention_impl = implementation
    return tc


def loss_and_gradient(mc, p, batch, implementation):
    tc = training_config(implementation)
    fn = jax.jit(jax.value_and_grad(
        lambda params, data: sample_losses(params, data, mc, tc), has_aux=True))
    return fn(p, batch)


def test_xla_valid_outputs_for_every_history_length(small_policy):
    """All 1..512 lengths, all bucket boundaries, and an unpadded 512 row."""
    mc, p = small_policy
    manual = jax.jit(lambda p, t, l: encode_history(p, t, l, mc))
    candidate = jax.jit(lambda p, t, l: encode_history(
        p, t, l, mc, attention_impl='xla'))
    previous = 0
    for width in (64, 128, 256, 512):
        for start in range(previous + 1, width + 1, 16):
            tokens, lengths = histories(range(start, start + 16), width)
            a_ctx, a_hidden = manual(p, tokens, lengths)
            b_ctx, b_hidden = candidate(p, tokens, lengths)
            valid = tokens != 0
            np.testing.assert_allclose(b_ctx, a_ctx, rtol=3e-5, atol=3e-5)
            np.testing.assert_allclose(np.asarray(b_hidden)[valid],
                                       np.asarray(a_hidden)[valid], rtol=3e-5, atol=3e-5)
        previous = width


def test_xla_causality_and_interior_pad_key_mask(small_policy):
    mc, p = small_policy
    tokens, lengths = histories([64, 64], 64)
    tokens[1, :20] = tokens[0, :20]
    tokens[:, [3, 9, 31]] = 0
    changed = dict(p, embedding=p['embedding'].at[0].set(40.))
    outputs = []
    for implementation in ('manual', 'xla'):
        fn = jax.jit(lambda pp: encode_history(
            pp, tokens, lengths, mc, attention_impl=implementation)[1])
        before, after = np.asarray(fn(p)), np.asarray(fn(changed))
        valid = tokens != 0
        # PAD keys cannot influence valid positions, even with huge embeddings.
        np.testing.assert_allclose(after[valid], before[valid], rtol=3e-5, atol=3e-5)
        # Future non-PAD tokens cannot influence the common prefix.
        np.testing.assert_allclose(before[0, :20], before[1, :20],
                                   rtol=3e-5, atol=3e-5)
        outputs.append(before)
    np.testing.assert_allclose(outputs[1][tokens != 0], outputs[0][tokens != 0],
                               rtol=3e-5, atol=3e-5)


def test_xla_losses_and_all_parameter_gradients(small_policy):
    mc, p = small_policy
    batch = loss_batch()
    (a_loss, a_metrics), a_grad = loss_and_gradient(mc, p, batch, 'manual')
    (b_loss, b_metrics), b_grad = loss_and_gradient(mc, p, batch, 'xla')
    assert a_loss.dtype == b_loss.dtype == np.dtype('float32')
    assert a_metrics.dtype == b_metrics.dtype == np.dtype('float32')
    np.testing.assert_allclose(b_metrics, a_metrics, rtol=3e-5, atol=2e-6)
    assert set(a_grad) == set(b_grad) == set(p)
    for name in p:
        assert a_grad[name].dtype == b_grad[name].dtype == np.dtype('float32')
        assert np.isfinite(np.asarray(b_grad[name])).all(), name
        np.testing.assert_allclose(b_grad[name], a_grad[name],
                                   rtol=2e-4, atol=2e-6, err_msg=name)


def test_xla_one_optimizer_step_preserves_state_and_batch(small_policy):
    mc, p = small_policy
    before = asdict(mc)
    single = loss_batch(lengths=(6, 10, 16), width=16)
    batch = {k: np.stack([v, v]) for k, v in single.items()}
    opt = init_optimizer(p)
    baseline = make_train_step(mc, training_config('manual', 2))(p, opt, batch)
    candidate = make_train_step(mc, training_config('xla', 2))(p, opt, batch)
    assert asdict(mc) == before
    assert int(baseline[1]['step']) == int(candidate[1]['step']) == 1
    assert float(baseline[2][-1]) == float(candidate[2][-1]) == 1.
    for a, b in zip(jax.tree_util.tree_leaves(baseline),
                    jax.tree_util.tree_leaves(candidate)):
        np.testing.assert_allclose(b, a, rtol=2e-4, atol=2e-6)


def test_xla_inference_fused_and_overflow_preserve_scores_and_choices(small_policy):
    mc, p = small_policy
    rng = np.random.default_rng(220)
    requests = [{'tokens': rng.integers(1, 23, n, dtype=np.int32),
                 'state': rng.random(STATE_DIM, dtype=np.float32),
                 'actions': rng.random((count, ACTION_DIM), dtype=np.float32),
                 'role': i % 3}
                for i, (n, count) in enumerate(((6, 9), (64, 11), (129, 2)))]
    baseline = Inference(mc, batch=2, action_chunk=16)
    candidate = Inference(mc, batch=2, action_chunk=16, fused=True,
                          attention_impl='xla')
    for actual, expected in zip(candidate.score(p, requests), baseline.score(p, requests)):
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)
    assert candidate.choose(p, requests) == baseline.choose(p, requests)
    assert candidate.last_stats['fused_batches'] == 1
    assert candidate.last_stats['legal_actions'] == 22


def test_cudnn_host_padding_guard():
    tokens, lengths = histories([1, 6, 64], 64)
    validate_attention_tokens(tokens, lengths)
    validate_attention_tokens(np.stack([tokens, tokens]), np.stack([lengths, lengths]))
    for changed_tokens, changed_lengths, message in (
            (tokens.copy(), lengths.astype(np.float32), 'integer'),
            (tokens.copy(), lengths[:-1], 'shapes'),
            (tokens.copy(), np.array([0, 6, 64], np.int32), 'nonempty'),
            (tokens.copy(), np.array([1, 6, 65], np.int32), 'nonempty')):
        with pytest.raises(ValueError, match=message):
            validate_attention_tokens(changed_tokens, changed_lengths)
    interior_pad = tokens.copy()
    interior_pad[1, 2] = 0
    suffix_token = tokens.copy()
    suffix_token[0, 10] = 3
    for invalid in (interior_pad, suffix_token):
        with pytest.raises(ValueError, match='right padding'):
            validate_attention_tokens(invalid, lengths)


def test_cudnn_rejects_interior_pad_before_dispatch(small_policy):
    mc, p = small_policy
    request = {'tokens': np.array([1, 2, 0, 3], np.int32),
               'state': np.zeros(STATE_DIM, np.float32), 'role': 0,
               'actions': np.ones((1, ACTION_DIM), np.float32)}
    with pytest.raises(ValueError, match='right padding'):
        Inference(mc, batch=1, fused=True, attention_impl='cudnn').choose(p, [request])


@pytest.mark.parametrize('fused', [False, True])
def test_cudnn_does_not_silently_fall_back_for_fp32(small_policy, fused):
    mc, p = small_policy
    request = {'tokens': np.array([1, 2, 8, 9, 10, 3], np.int32),
               'state': np.zeros(STATE_DIM, np.float32), 'role': 0,
               'actions': np.ones((1, ACTION_DIM), np.float32)}
    with pytest.raises(ValueError, match='requires ModelConfig.bf16'):
        Inference(mc, batch=1, fused=fused, attention_impl='cudnn').choose(p, [request])


def test_attention_rejects_unknown_implementation(small_policy):
    mc, p = small_policy
    tokens, lengths = histories([6], 64)
    with pytest.raises(ValueError, match='attention_impl'):
        Inference(mc, attention_impl='auto')
    with pytest.raises(ValueError, match='attention_impl'):
        encode_history(p, tokens, lengths, mc, attention_impl='auto')


@pytest.mark.skipif(any(d.platform == 'gpu' for d in jax.devices()),
                    reason='This test verifies failure on a CPU-only backend')
def test_cudnn_bf16_requires_cuda_backend():
    mc = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16,
                     bf16=True, remat=False)
    p = init_params(mc, 81)
    tokens, lengths = histories([6], 64)
    fn = jax.jit(lambda pp, t, l: encode_history(
        pp, t, l, mc, attention_impl='cudnn'))
    with pytest.raises((RuntimeError, NotImplementedError), match='cuDNN|cudnn|platform'):
        jax.block_until_ready(fn(p, tokens, lengths))


def relative_l2(actual, expected):
    a, b = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-7)


@pytest.mark.skipif(not any(d.platform == 'gpu' for d in jax.devices()),
                    reason='cuDNN parity needs a real CUDA GPU; CPU is not GPU validation')
def test_cudnn_bf16_valid_outputs_losses_and_gradients():
    """Actual model dimensions, ragged padding, and a fully occupied 512 row.

    BF16 fused reduction order differs from the manual attention. Bound valid
    activation/gradient vector errors, rather than requiring bitwise equality
    or unstable per-element relative errors at values near zero.
    """
    mc = ModelConfig(remat=False)
    p = init_params(mc, 531)
    batch = loss_batch(lengths=(1, 6, 129, 512), width=512)
    validate_attention_tokens(batch['tokens'], batch['lengths'])
    valid = batch['tokens'] != 0
    histories_out = []
    for implementation in ('manual', 'cudnn'):
        fn = jax.jit(lambda pp, t, l: encode_history(
            pp, t, l, mc, attention_impl=implementation))
        histories_out.append(jax.device_get(fn(p, batch['tokens'], batch['lengths'])))
    assert relative_l2(histories_out[1][0], histories_out[0][0]) < .04
    assert relative_l2(histories_out[1][1][valid], histories_out[0][1][valid]) < .04
    (a_loss, a_metrics), a_grad = loss_and_gradient(mc, p, batch, 'manual')
    (b_loss, b_metrics), b_grad = loss_and_gradient(mc, p, batch, 'cudnn')
    assert a_loss.dtype == b_loss.dtype == np.dtype('float32')
    np.testing.assert_allclose(b_metrics, a_metrics, rtol=.03, atol=5e-4)
    for gradient in (a_grad, b_grad):
        assert all(g.dtype == np.dtype('float32') for g in gradient.values())
    a_vector = np.concatenate([np.asarray(a_grad[k]).ravel() for k in sorted(p)])
    b_vector = np.concatenate([np.asarray(b_grad[k]).ravel() for k in sorted(p)])
    assert relative_l2(b_vector, a_vector) < .08
