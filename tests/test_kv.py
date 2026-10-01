import numpy as np
import pytest

from dougpu.config import ModelConfig
from dougpu.model import init_params
from dougpu.inference import Inference
from dougpu.kv_inference import KVInference


def request(tokens, role=0):
    return {'tokens': np.asarray(tokens, np.uint8), 'state': np.linspace(0., .8, 21, dtype=np.float32),
            'role': role, 'actions': np.arange(7*16, dtype=np.float32).reshape(7, 16)/112}


@pytest.mark.parametrize('bf16', [False, True])
def test_incremental_prefix_offsets_buckets_reset_and_versions(bf16):
    mc = ModelConfig(width=32, layers=2, heads=2, ffn=64, q_hidden=32, bf16=bf16)
    p = init_params(mc, 19)
    full = Inference(mc, batch=4, action_chunk=32, bucket_batch=True)
    cached = KVInference(mc, capacity=4, batch=4, action_chunk=32, bucket_batch=True)
    rng = np.random.default_rng(84)
    histories = [rng.integers(1, 23, 512, dtype=np.uint8) for _ in range(4)]
    longest = 0.
    for length, keys in [(6, [0, 1, 2]), (9, [2, 0]), (63, [0, 1]), (64, [0, 2]),
                          (66, [1, 0]), (127, [2, 1]), (129, [0, 2, 1]),
                          (256, [1, 2]), (257, [0, 1]), (512, [2, 0])]:
        reqs = [request(histories[k][:length], k%3) for k in keys]
        expected = full.score(p, reqs)
        actual = cached.score(p, reqs, cache_keys=keys, policy_version=1)
        for a, b in zip(expected, actual):
            longest = max(longest, float(np.max(np.abs(a-b))))
            np.testing.assert_allclose(a, b, atol=3e-4 if bf16 else 3e-7, rtol=.02 if bf16 else 2e-5)
        # An unchanged prefix reuses the final public context without appending.
        repeated = cached.score(p, reqs, cache_keys=keys, policy_version=1)
        assert cached.cache_stats['appended_tokens'] == 0
        for a, b in zip(actual, repeated):
            np.testing.assert_array_equal(a, b)
    changed = request([1, 2, 21, 8, 19, 3])
    a = full.score(p, [changed])[0]
    b = cached.score(p, [changed], cache_keys=[0], policy_version=1)[0]
    assert cached.cache_stats['reset_slots'] == 1
    np.testing.assert_allclose(a, b, atol=3e-4 if bf16 else 3e-7, rtol=.02 if bf16 else 2e-5)
    new = init_params(mc, 23)
    for params, version in [(new, 1), (new, 2), (p, 3)]:
        expected = full.score(params, [changed])[0]
        actual = cached.score(params, [changed], cache_keys=[0], policy_version=version)[0]
        assert cached.cache_stats['reused_tokens'] == 0
        np.testing.assert_allclose(expected, actual, atol=3e-4 if bf16 else 3e-7, rtol=.02 if bf16 else 2e-5)
    print('bf16', bf16, 'max_q_abs', longest)


def test_cached_choice_overflow_rng_ties_finite_and_key_validation():
    mc = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16, bf16=False)
    p = init_params(mc, 52)
    full = Inference(mc, batch=2, action_chunk=31)
    cached = KVInference(mc, capacity=3, batch=2, action_chunk=31)
    reqs = [request([1, 2, 8, 9, 10, 3]), request([1, 2, 9, 9, 10, 3], 2)]
    reqs[0]['actions'] = np.tile(reqs[0]['actions'], (13, 1))
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    assert full.choose(p, reqs, a, .5) == cached.choose(p, reqs, b, .5, cache_keys=[2, 0], policy_version=0)
    assert a.bit_generator.state == b.bit_generator.state
    for keys in ([0, 0], [-1, 1], [1, 3], [1], [1.1, 2.1]):
        with pytest.raises(ValueError):
            cached.choose(p, reqs, cache_keys=keys, policy_version=0)
    tied = dict(p, **{'qout.w': np.zeros_like(p['qout.w']), 'qout.b': np.zeros_like(p['qout.b'])})
    assert cached.choose(tied, reqs, cache_keys=[2, 0], policy_version=1) == [0, 0]
    bad = dict(p, **{'qout.b': np.full(3, np.nan, np.float32)})
    with pytest.raises(FloatingPointError):
        cached.choose(bad, reqs, cache_keys=[2, 0], policy_version=2)
    assert cached.cache is None
    assert cached.choose(p, reqs, cache_keys=[2, 0], policy_version=3) == full.choose(p, reqs)


def test_frozen_cached_evaluation_matches_serial_and_parallel():
    from dougpu.actors import ActorPool
    from dougpu.config import TrainConfig
    from dougpu.evaluation import paired_evaluate
    mc = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16, bf16=False)
    p, opponent = init_params(mc, 11), init_params(mc, 13)
    infer = Inference(mc, batch=8, action_chunk=127, bucket_batch=True)
    pool = ActorPool(TrainConfig(engine='reference', workers=2, envs_per_worker=1), [17, 18])
    before = [r['tokens'].copy() for r in pool.requests]
    try:
        for other in ('rule', 'random', opponent):
            expected = paired_evaluate(p, other, infer, 'reference', 3, 122)
            assert expected == paired_evaluate(p, other, infer, 'reference', 3, 122, kv_cache=True)
            assert expected == paired_evaluate(p, other, infer, 'reference', 3, 122,
                                               actor_pool=pool, workers=2, kv_cache=True)
        for old, req in zip(before, pool.requests):
            np.testing.assert_array_equal(old, req['tokens'])
    finally:
        pool.close()
