from dataclasses import asdict, replace
import jax
import numpy as np
import pytest
from dougpu.config import ModelConfig, TrainConfig
from dougpu.inference import Inference, merge_inference_stats
from dougpu.model import init_params, init_optimizer, make_train_step
from test_core import train_batch


def requests():
    rng = np.random.default_rng(819)
    lengths = [257, 6, 64, 129, 65, 128, 256, 512] * 8
    return [{'tokens': rng.integers(1, 23, n, dtype=np.int32),
             'state': rng.random(21, dtype=np.float32), 'role': i % 3,
             'actions': rng.random((1+i % 5, 16), dtype=np.float32)}
            for i, n in enumerate(lengths)]


@pytest.fixture(scope='module')
def policy():
    mc = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16, bf16=False)
    return mc, init_params(mc, 83)


@pytest.mark.parametrize('history,fused', [(True, False), (False, True), (True, True)])
def test_order_rng_scores_ties_and_overflow(policy, history, fused):
    mc, p = policy
    req = requests()
    baseline = Inference(mc, batch=128, action_chunk=256)
    candidate = Inference(mc, batch=128, action_chunk=256, history_buckets=history, fused=fused)
    expected_scores = baseline.score(p, req)
    actual_scores = candidate.score(p, req)
    for a, b in zip(expected_scores, actual_scores):
        np.testing.assert_allclose(a, b, atol=2e-7, rtol=5e-5)
    for epsilon in (0., .4, 1.):
        a, b = np.random.default_rng(92), np.random.default_rng(92)
        assert baseline.choose(p, req, a, epsilon) == candidate.choose(p, req, b, epsilon)
        assert a.bit_generator.state == b.bit_generator.state
    if history:
        assert candidate.last_stats['history_slots'] < baseline.last_stats['history_slots']
    if fused:
        assert candidate.last_stats['fused_batches'] > 0
    tied = dict(p, **{'qout.w': np.zeros_like(p['qout.w']), 'qout.b': np.zeros(3, np.float32)})
    req[0]['actions'] = np.tile(req[0]['actions'], (513, 1))
    assert candidate.choose(tied, req) == [0] * len(req)
    assert baseline.choose(p, req) == candidate.choose(p, req)
    assert candidate.last_stats['legal_actions'] == sum(len(r['actions']) for r in req)
    assert candidate.choose(p, []) == []
    assert candidate.score(p, []) == []
    assert candidate.last_stats['requests'] == 0


def test_partition_cost_guard_and_invalid_inputs(policy):
    mc, p = policy
    inf = Inference(mc, 128, 256, history_buckets=True, fused=True)
    req = requests()
    plans = list(inf._plans(req + req + req[:7]))
    ids = np.concatenate([i for i, _, _ in plans])
    np.testing.assert_array_equal(np.sort(ids), np.arange(135))
    assert all(b in (32, 64, 128) and t in (64, 128, 256, 512) for _, b, t in plans)
    small = Inference(mc, 8, 256, bucket_batch=True, history_buckets=True)
    assert len(list(small._plans(req[:8]))) == 1
    for bad in (0, 513):
        with pytest.raises(ValueError, match='history'):
            inf.choose(p, [dict(req[0], tokens=np.ones(bad, np.int32))])
    with pytest.raises(ValueError, match='legal action'):
        inf.choose(p, [dict(req[0], actions=np.zeros((0, 16), np.float32))])
    rng = np.random.default_rng(3)
    saved = rng.bit_generator.state
    for bad in (np.nan, np.inf, -np.inf):
        broken = dict(p, **{'qout.b': np.full(3, bad, np.float32)})
        with pytest.raises(FloatingPointError):
            inf.choose(broken, req[:1], rng, .5)
    assert rng.bit_generator.state == saved


def test_weighted_padding_not_mean_of_fractions():
    rows = [{'requests': 1, 'state_slots': 32, 'history_tokens': 6, 'history_slots': 2048,
             'legal_actions': 1, 'action_slots': 256, 'max_legal_actions': 1, 'shapes': {'32x64': 1}},
            {'requests': 128, 'state_slots': 128, 'history_tokens': 16000, 'history_slots': 16384,
             'legal_actions': 500, 'action_slots': 512, 'max_legal_actions': 8, 'shapes': {'128x128': 1}}]
    stats = merge_inference_stats(rows)
    assert stats['history_padding_fraction'] == 1 - 16006/18432
    assert stats['action_padding_fraction'] == 1 - 501/768
    assert stats['max_legal_actions'] == 8 and sum(stats['shapes'].values()) == 2


def test_runtime_remat_preserves_model_config_and_optimizer(policy):
    mc, p = policy
    tc = TrainConfig(micro_batch=3, accumulation=2)
    before = asdict(mc)
    opt = init_optimizer(p)
    batch = train_batch()
    a = make_train_step(mc, tc)(p, opt, batch)
    b = make_train_step(mc, replace(tc, learner_remat=False))(p, opt, batch)
    assert asdict(mc) == before and mc.remat is True
    assert int(a[1]['step']) == int(b[1]['step']) == 1
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x, y, rtol=5e-5, atol=5e-7)
    with pytest.raises(ValueError, match='learner_remat'):
        replace(tc, learner_remat='false').validate()
    with pytest.raises(ValueError, match='KV cache'):
        replace(tc, infer_history_buckets=True, selfplay_kv_cache=True).validate()
    with pytest.raises(ValueError, match='KV cache'):
        replace(tc, eval_optimized_inference=True, eval_kv_cache=True).validate()
