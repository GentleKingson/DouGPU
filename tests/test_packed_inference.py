from copy import deepcopy
import numpy as np
import pytest

from dougpu.actors import ActorPool, Ring
from dougpu.config import ModelConfig, TrainConfig
from dougpu.encoding import PackedRequests, pack_requests
from dougpu.inference import Inference
from dougpu.model import init_params
from dougpu.replay import Replay


def corpus():
    rng = np.random.default_rng(149)
    lengths = [6, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 8]
    return [{'tokens': rng.integers(1, 23, n, dtype=np.uint8),
             'state': rng.random(21, dtype=np.float32), 'role': i % 3,
             'actions': rng.random((33 if i == 3 else 1+i % 5, 16), dtype=np.float32)}
            for i, n in enumerate(lengths)]


def packed(rows):
    return PackedRequests.merge([pack_requests(rows[i:i+3]) for i in range(0, len(rows), 3)])


def assert_request(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


@pytest.mark.parametrize('batch,bucket,history', [(5, False, False), (8, True, False),
                                               (128, False, False), (256, False, False),
                                               (128, False, True)])
def test_packed_slices_and_exact_inference_tensors(batch, bucket, history):
    rows = corpus()
    req = packed(rows)
    infer = Inference(ModelConfig(), batch, 16, bucket_batch=bucket, history_buckets=history)
    for selection in (slice(None), slice(2, 11), slice(4, 4), slice(10, 2), slice(-7, -1),
                      slice(0, 50, 2), slice(None, None, -1)):
        a, b = rows[selection], req[selection]
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_request(x, y)
        want = list(infer._batches(a))
        stats = deepcopy(infer.last_stats)
        got = list(infer._batches(b))
        assert stats == infer.last_stats and len(want) == len(got)
        for x, y in zip(want, got):
            for u, v in zip(x, y):
                np.testing.assert_array_equal(u, v)
                assert u.dtype == v.dtype
    assert_request(rows[-1], req[-1])
    with pytest.raises(IndexError):
        req[len(req)]
    assert PackedRequests.merge([]) == []
    # A view keeps its old arrays alive when the pool replaces the next round.
    view = req[2:7]
    new = packed(corpus())
    new.data['tokens'][:] = 0
    for x, y in zip(rows[2:7], view):
        assert_request(x, y)


@pytest.mark.parametrize('bf16,fused', [(False, False), (True, False), (True, True)])
def test_packed_choices_scores_rng_ties_and_invalid_values(bf16, fused):
    mc = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16, bf16=bf16)
    p = init_params(mc, 63)
    rows = corpus()
    req = packed(rows)
    infer = Inference(mc, batch=8, action_chunk=16, fused=fused)
    for a, b in zip(infer.score(p, rows), infer.score(p, req)):
        np.testing.assert_array_equal(a, b)
    for epsilon in (0., .4, 1.):
        a, b = np.random.default_rng(94), np.random.default_rng(94)
        assert infer.choose(p, rows, a, epsilon) == infer.choose(p, req, b, epsilon)
        assert a.bit_generator.state == b.bit_generator.state
    tied = dict(p, **{'qout.w': np.zeros_like(p['qout.w']), 'qout.b': np.zeros(3, np.float32)})
    assert infer.choose(tied, req) == [0] * len(rows)
    assert infer.choose(p, req[:0]) == []
    rng = np.random.default_rng(42)
    before = deepcopy(rng.bit_generator.state)
    for bad in (np.nan, np.inf, -np.inf):
        broken = dict(p, **{'qout.b': np.full(3, bad, np.float32)})
        with pytest.raises(FloatingPointError):
            infer.choose(broken, req, rng, .4)
        assert rng.bit_generator.state == before
    for n in (0, 513):
        invalid = [dict(rows[0], tokens=np.ones(n, np.uint8))]
        with pytest.raises(ValueError, match='history'):
            infer.choose(p, packed(invalid))
    invalid = [dict(rows[0], actions=np.zeros((0, 16), np.float32))]
    with pytest.raises(ValueError, match='legal action'):
        infer.choose(p, packed(invalid))


def test_actor_packed_path_preserves_complete_rounds_without_unpack(monkeypatch):
    import dougpu.actors as actors
    monkeypatch.setattr(actors, 'unpack_requests', lambda *_: pytest.fail('Packed path unpacked'))
    cfg = TrainConfig(engine='reference', workers=2, envs_per_worker=2)
    seeds = [152, 153]
    pool = ActorPool(cfg, seeds, packed=True)
    rings = [Ring('reference', 2, seed) for seed in seeds]
    expected, actual = Replay(197), Replay(197)
    rng = np.random.default_rng(53)
    try:
        for version in range(120):
            req = [r for ring in rings for r in ring.requests()]
            assert isinstance(pool.requests, PackedRequests)
            for a, b in zip(req, pool.requests):
                assert_request(a, b)
            choices = [int(rng.integers(len(r['actions']))) for r in req]
            rows, wins = [], []
            for i, ring in enumerate(rings):
                samples, winners = ring.advance(choices[2*i:2*i+2], version)
                rows.extend(samples)
                wins.extend(winners)
                pool.submit(choices[2*i:2*i+2], version, i, i+1)
            batch, got = pool.receive()
            assert wins == got
            expected.add(rows)
            actual.add(batch)
        assert (expected.pos, expected.size) == (actual.pos, actual.size)
        for k in expected.data:
            np.testing.assert_array_equal(expected.data[k], actual.data[k])
    finally:
        pool.close()


def test_resume_does_not_initialize_discarded_parameters(tmp_path, monkeypatch):
    import json
    import sys
    import jax
    import dougpu.model as model
    from dougpu.checkpoint import Store
    from dougpu.train import main
    cfg = {'model': {'width': 16, 'layers': 1, 'heads': 2, 'ffn': 32, 'q_hidden': 16},
           'train': {'engine': 'reference', 'require_tpu': False,
                     'backend': 'cuda' if jax.default_backend() == 'gpu' else jax.default_backend(), 'workers': 1,
                     'envs_per_worker': 1, 'max_cycles': 1, 'max_hours': 1e-9}}
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(path), '--workdir', str(tmp_path/'run')])
    main()
    before = Store(tmp_path/'run/checkpoints').load_latest()
    assert before is not None
    monkeypatch.setattr(model, 'init_params', lambda *_: pytest.fail('Unused initialization on resume'))
    main()
    after = Store(tmp_path/'run/checkpoints').load_latest()
    for k in before['params']:
        np.testing.assert_array_equal(before['params'][k], after['params'][k])
