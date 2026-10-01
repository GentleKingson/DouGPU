from copy import deepcopy
import jax
import numpy as np
import pytest

from dougpu.config import ModelConfig
from dougpu.encoding import PackedRequests, pack_requests
from dougpu.inference import Inference, _merge_best
from dougpu.model import init_params


def requests():
    rng = np.random.default_rng(702)
    return [{'tokens': rng.integers(1, 23, length, dtype=np.uint8),
             'state': rng.random(21, dtype=np.float32), 'role': i % 3,
             'actions': rng.random((count, 16), dtype=np.float32)}
            for i, (length, count) in enumerate(zip(
                (6, 64, 65, 128, 129, 256, 257, 512),
                (1, 1023, 1024, 1025, 2047, 2048, 2049, 3)))]


@pytest.mark.parametrize('bf16', [False, True])
def test_chunk_sizes_scores_choices_rng_and_all_history_buckets(bf16):
    mc = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16, bf16=bf16)
    params = init_params(mc, 703)
    rows = requests()
    packed = PackedRequests.merge([pack_requests(rows)])
    baseline, candidate = (Inference(mc, 2, chunk) for chunk in (2048, 1024))
    for expected, actual in zip(baseline.score(params, packed), candidate.score(params, packed)):
        # Different GEMM shapes need numerical parity, not bitwise identity.
        np.testing.assert_allclose(expected, actual, rtol=5e-7, atol=1e-9)
    for epsilon in (0., .4, 1.):
        a, b = (np.random.default_rng(704) for _ in range(2))
        assert baseline.choose(params, packed, a, epsilon) == candidate.choose(params, packed, b, epsilon)
        assert a.bit_generator.state == b.bit_generator.state
    tied = dict(params, **{'qout.w': np.zeros_like(params['qout.w']), 'qout.b': np.zeros(3, np.float32)})
    assert candidate.choose(tied, packed) == [0] * len(rows)
    assert candidate.choose(params, []) == []
    assert candidate.score(params, []) == []


@pytest.mark.parametrize('chunk', [1024, 2048])
def test_overflow_winner_tie_and_nonfinite_after_first_chunk(chunk):
    mc = ModelConfig()
    infer = Inference(mc, 4, chunk)
    infer.encoder = lambda p, tokens, lengths, state: np.zeros((len(tokens), 1), np.float32)
    infer.selector = jax.jit(lambda p, ctx, actions, owner, roles, valid, offset, best, indices, finite:
        _merge_best(actions[:, 0], owner, valid, offset, best, indices, finite))
    rows = [{'tokens': np.array([1], np.uint8), 'state': np.zeros(21, np.float32),
             'role': i, 'actions': np.zeros((n, 16), np.float32)}
            for i, n in enumerate((1025, 1, 2049))]
    rows[0]['actions'][1023:1025, 0] = 3
    rows[2]['actions'][-1, 0] = 4
    assert infer.choose({}, rows) == [1023, 0, 2048]
    assert infer.last_stats['legal_actions'] == 3075
    rows[0]['actions'][1024, 0] = np.nextafter(np.float32(3), np.float32(4))
    assert infer.choose({}, rows) == [1024, 0, 2048]
    for bad in (np.nan, np.inf, -np.inf):
        rows[2]['actions'][-1, 0] = bad
        rng = np.random.default_rng(705)
        before = deepcopy(rng.bit_generator.state)
        with pytest.raises(FloatingPointError):
            infer.choose({}, rows, rng, .4)
        assert rng.bit_generator.state == before
