import json
from dataclasses import asdict
import numpy as np
import pytest
from dougpu.config import ModelConfig, TrainConfig
from dougpu.encoding import encode, oracle_label, terminal_target, PublicState
from dougpu.reference import ReferenceGame
from dougpu.actors import Ring, ActorPool
from dougpu.replay import Replay
from dougpu.checkpoint import Store, load_policy


def test_reward_and_information_boundary():
    for winner in range(3):
        r = [terminal_target(winner, i) for i in range(3)]
        assert r[1] == r[2] == -r[0]
    game = ReferenceGame(np.random.default_rng(4))
    before = encode(game.public())
    label = oracle_label(game.oracle(), game.role)
    # Swap two hidden cards while preserving public counts and acting hand.
    h = game.hands
    i, j = next((i, j) for i, x in enumerate(h[1]) for j, y in enumerate(h[2]) if x != y)
    h[1][i], h[2][j] = h[2][j], h[1][i]
    after = encode(game.public())
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])
    assert not np.array_equal(label, oracle_label(game.oracle(), game.role))
    assert set(before) == {'tokens', 'state', 'role', 'actions'}


def test_ring_replay_and_balance():
    ring = Ring('reference', 3, 6)
    replay = Replay(128)
    rng = np.random.default_rng(6)
    for _ in range(120):
        req = ring.requests()
        samples, _ = ring.advance([int(rng.integers(len(r['actions']))) for r in req], 7)
        replay.add(samples)
        if replay.size == 128:
            break
    assert replay.size == 128
    tc = TrainConfig(micro_batch=2, accumulation=4)
    b = replay.sample(tc, rng, 7)
    roles, w = b['role'].reshape(-1), b['weight'].reshape(-1)
    for r in range(3):
        np.testing.assert_allclose(w[roles == r].sum(), 8/3, rtol=1e-6)
    restored = Replay(128)
    restored.restore(replay.export())
    assert restored.pos == replay.pos
    for k in replay.data:
        np.testing.assert_array_equal(replay.data[k], restored.data[k])
    assert len(replay.eligible(100, 1)) == 0


def test_spawn_actors():
    cfg = TrainConfig(engine='reference', workers=1, envs_per_worker=2)
    pool = ActorPool(cfg, [9])
    try:
        assert len(pool.requests) == 2
        for _ in range(3):
            pool.step([0, 0], 0)
    finally:
        pool.close()
    assert all(not p.is_alive() for p in pool.processes)


@pytest.fixture(scope='module')
def tiny():
    import jax
    from dougpu.model import init_params
    cfg = ModelConfig(width=32, layers=1, heads=2, ffn=64, q_hidden=32, bf16=False)
    return cfg, init_params(cfg, 1)


def test_causal_and_padding(tiny):
    from dougpu.model import encode_history
    cfg, params = tiny
    tokens = np.zeros((2, 16), np.int32)
    tokens[:, :12] = np.arange(1, 13)
    tokens[1, 8:12] = [16, 18, 15, 19]
    _, hidden = encode_history(params, tokens, np.array([12, 12]), cfg)
    np.testing.assert_allclose(np.asarray(hidden[0, :8]), np.asarray(hidden[1, :8]), atol=2e-5)
    a, _ = encode_history(params, tokens[:1], np.array([12]), cfg)
    b, _ = encode_history(params, np.pad(tokens[:1], ((0, 0), (0, 16))), np.array([12]), cfg)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=2e-5)


def test_chunked_candidates_not_truncated(tiny):
    from dougpu.inference import Inference
    cfg, params = tiny
    req = encode(ReferenceGame(np.random.default_rng(1)).public())
    # Artificially duplicate candidates to force many chunks.
    req['actions'] = np.tile(req['actions'], (3, 1))
    a = Inference(cfg, batch=2, action_chunk=7).score(params, [req])[0]
    b = Inference(cfg, batch=2, action_chunk=4096).score(params, [req])[0]
    assert len(a) == len(req['actions']) > 7
    np.testing.assert_allclose(a, b, atol=1e-5)


def train_batch(seq=16):
    b = {'tokens': np.zeros((2, 3, seq), np.int32), 'lengths': np.full((2, 3), 6, np.int32),
         'state': np.ones((2, 3, 21), np.float32)*.2,
         'actions': np.ones((2, 3, 16), np.float32)*.25,
         'role': np.tile(np.arange(3, dtype=np.int32), (2, 1)),
         'belief': np.ones((2, 3, 30), np.float32)*.2,
         'target': np.array([[1., -1., -1.], [1., -1., -1.]], np.float32),
         'weight': np.ones((2, 3), np.float32)}
    b['tokens'][:, :, :6] = [1, 2, 8, 9, 10, 3]
    return b


def test_optimizer_updates_and_skips_nonfinite(tiny):
    from dougpu.model import make_train_step, init_optimizer
    cfg, params = tiny
    tc = TrainConfig(micro_batch=3, accumulation=2)
    step = make_train_step(cfg, tc)
    opt = init_optimizer(params)
    batch = train_batch()
    updated, opt2, metrics = step(params, opt, batch)
    assert np.asarray(metrics)[-1] == 1 and int(opt2['step']) == 1
    assert any(not np.array_equal(params[k], updated[k]) for k in params)
    batch['target'][:] = np.nan
    skipped, opt3, metrics = step(updated, opt2, batch)
    assert np.asarray(metrics)[-1] == 0 and int(opt3['step']) == 1
    for k in updated:
        np.testing.assert_array_equal(updated[k], skipped[k])


def test_checkpoint_fallback_and_policy(tmp_path, tiny):
    from dougpu.model import init_optimizer
    cfg, params = tiny
    replay = Replay(8)
    store = Store(tmp_path/'local', tmp_path/'remote', keep=3)
    meta = {'cycle': 1, 'updates': 1, 'model': asdict(cfg), 'champion_cycle': 0}
    first = store.save(params, init_optimizer(params), params, replay, meta)
    second = store.save(params, init_optimizer(params), params, replay, dict(meta, cycle=2))
    assert store.last_remote_ok
    for path in (second, tmp_path/'remote'/second.name):
        path.write_bytes(b'broken')
    loaded = store.load_latest()
    assert loaded['meta']['cycle'] == 1
    for k in params:
        np.testing.assert_array_equal(loaded['params'][k], params[k])
    exported, policy_meta = load_policy(tmp_path/'local'/'latest_policy.npz')
    assert policy_meta['model'] == asdict(cfg)
    assert 'ntp' not in exported and not any(k.startswith('belief') for k in exported)


def test_bf16_train_and_disabled_auxiliary():
    from dougpu.model import init_params, init_optimizer, make_train_step
    cfg = ModelConfig(width=32, layers=1, heads=2, ffn=64, q_hidden=32, bf16=True)
    tc = TrainConfig(micro_batch=3, accumulation=2, ntp_weight=0., belief_weight=0.)
    p = init_params(cfg, 3)
    p, opt, metrics = make_train_step(cfg, tc)(p, init_optimizer(p), train_batch())
    m = np.asarray(metrics)
    assert np.isfinite(m).all() and m[-1] == 1 and int(opt['step']) == 1
    assert m[2] == 0 and m[3] == 0


def test_default_chunk_overflow_and_multiple_owners(tiny):
    from dougpu.inference import Inference
    cfg, params = tiny
    game = ReferenceGame(np.random.default_rng(19))
    first = encode(game.public())
    game.step(0)
    second = encode(game.public())
    first['actions'] = np.tile(first['actions'], (4101//len(first['actions'])+1, 1))[:4101]
    a = Inference(cfg, batch=2, action_chunk=2048).score(params, [first, second])
    b = Inference(cfg, batch=2, action_chunk=8192).score(params, [first, second])
    assert len(a[0]) == 4101 and len(a[1]) == len(second['actions'])
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=1e-5)
