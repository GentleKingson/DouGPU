from dataclasses import replace
import pickle
import numpy as np
import pytest
from dougpu.config import ModelConfig, TrainConfig
from dougpu.encoding import encode, tokenize, pack_requests, unpack_requests, oracle_label
from dougpu.environment import DouZeroGame
from dougpu.actors import ActorPool, Ring
from dougpu.replay import Replay, PackedSamples


def same_request(a, b):
    assert set(a) == set(b) == {'tokens', 'state', 'role', 'actions'}
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_compact_rules_and_incremental_tokens_match_upstream():
    pytest.importorskip('douzero.env.game')
    for seed in range(32):
        rng = np.random.default_rng(seed + 901)
        fast = DouZeroGame(np.random.default_rng(seed))
        full = DouZeroGame(np.random.default_rng(seed), compact=False)
        snapshots = []
        while not fast.done:
            a, b = fast.public(), full.public()
            assert a == b
            encoded = encode(a)
            np.testing.assert_array_equal(encoded['tokens'], tokenize(replace(a, history_tokens=None)))
            same_request(encoded, encode(replace(b, history_tokens=None)))
            snapshots.append((a, encoded))
            assert fast.full_infoset().__dict__ == full.game.game_infoset.__dict__
            for x, y in zip(fast.oracle(), full.oracle()):
                assert x == y
            index = int(rng.integers(len(a.legal))) if seed % 2 else max(
                range(len(a.legal)), key=lambda i: len(a.legal[i]))
            fast.step(index)
            full.step(index)
            assert fast.done == full.done and fast.winner == full.winner
        for public, saved in snapshots:
            same_request(encode(public), saved)
        assert fast.game.bomb_num == full.game.bomb_num
        assert fast.game.player_utility_dict == full.game.player_utility_dict


def test_compact_oracle_is_not_a_policy_input():
    pytest.importorskip('douzero.env.game')
    game = DouZeroGame(np.random.default_rng(19))
    before = encode(game.public())
    hands = game.oracle()
    label = oracle_label(hands, game.role)
    i, j = next((i, j) for i, x in enumerate(hands[1]) for j, y in enumerate(hands[2]) if x != y)
    hands[1][i], hands[2][j] = hands[2][j], hands[1][i]
    same_request(before, encode(game.public()))
    assert not np.array_equal(label, oracle_label(hands, game.role))


def test_packed_requests_and_actor_protocol():
    cfg = TrainConfig(engine='reference', workers=2, envs_per_worker=2)
    rings = [Ring('reference', 2, s) for s in (21, 22)]
    pool = ActorPool(cfg, [21, 22])
    rng = np.random.default_rng(77)
    expected_replay, actual_replay = Replay(197), Replay(197)
    try:
        for step in range(120):
            requests = [r for ring in rings for r in ring.requests()]
            restored = unpack_requests(pickle.loads(pickle.dumps(pack_requests(requests))))
            for a, b, c in zip(requests, pool.requests, restored):
                same_request(a, b)
                same_request(a, c)
            choices = [int(rng.integers(len(r['actions']))) for r in requests]
            rows, winners = [], []
            for i, ring in enumerate(rings):
                samples, wins = ring.advance(choices[2*i:2*i+2], step // 10)
                rows.extend(samples)
                winners.extend(wins)
            packed, actual_wins = pool.step(choices, step // 10)
            assert len(packed) == len(rows) and actual_wins == winners
            expected_replay.add(rows)
            actual_replay.add(packed)
        for k in expected_replay.data:
            np.testing.assert_array_equal(expected_replay.data[k], actual_replay.data[k])
    finally:
        pool.close()
    assert all(not p.is_alive() for p in pool.processes)


@pytest.mark.parametrize('capacity', [1, 3, 17])
def test_bulk_replay_wrap_and_oversized_batch(capacity):
    actual = Replay(capacity)
    expected = Replay(capacity)
    for n in (0, 1, 3, 5, 37, 0, 2):
        rows = [(np.arange(1, i % 16 + 2, dtype=np.uint8), np.full(21, i / 7),
                 np.full(16, i / 9), i % 3, np.full(30, i / 11), float(i % 2), i)
                for i in range(n)]
        for t, s, a, r, b, y, v in rows:
            j = expected.pos
            expected.data['tokens'][j] = 0
            expected.data['tokens'][j, :len(t)] = t
            for k, val in [('lengths', len(t)), ('state', s), ('actions', a), ('role', r),
                           ('belief', b), ('target', y), ('version', v)]:
                expected.data[k][j] = val
            expected.pos = (j + 1) % capacity
            expected.size = min(capacity, expected.size + 1)
        actual.add(PackedSamples.pack(rows))
        assert (actual.pos, actual.size) == (expected.pos, expected.size)
        for k in actual.data:
            np.testing.assert_array_equal(actual.data[k], expected.data[k])


def test_device_reduction_ties_overflow_and_nonfinite():
    import jax
    from dougpu.inference import _merge_best
    fn = jax.jit(_merge_best)
    best = np.full(3, -np.inf, np.float32)
    indices = np.full(3, np.iinfo(np.int32).max, np.int32)
    q = np.array([-2., -1., -1., 4., 4., -9., np.nan], np.float32)
    owner = np.array([0, 0, 0, 1, 1, 2, 0], np.int32)
    valid = np.arange(7) < 6
    best, indices, finite = fn(q, owner, valid, np.int32(0), best, indices, np.bool_(True))
    np.testing.assert_array_equal(indices, [1, 3, 5])
    assert bool(finite)
    best, indices, finite = fn(q, owner, valid, np.int32(7), best, indices, finite)
    np.testing.assert_array_equal(indices, [1, 3, 5])
    for bad in [np.nan, np.inf, -np.inf]:
        broken = q.copy()
        broken[0] = bad
        assert not bool(fn(broken, owner, valid, np.int32(14), best, indices, finite)[2])


@pytest.fixture(scope='module')
def policy():
    from dougpu.model import init_params
    mc = ModelConfig(width=32, layers=1, heads=2, ffn=64, q_hidden=32, bf16=False)
    return mc, init_params(mc, 48)


def test_choose_matches_score_and_rng(policy):
    from dougpu.inference import Inference
    from dougpu.reference import ReferenceGame
    mc, params = policy
    requests = [encode(ReferenceGame(np.random.default_rng(i)).public()) for i in range(5)]
    requests[0]['actions'] = np.tile(requests[0]['actions'], (100, 1))[:4101]
    requests[1]['actions'] = requests[1]['actions'][:1]
    infer = Inference(mc, batch=3, action_chunk=127)
    scores = infer.score(params, requests)
    for epsilon in (0., .5, 1.):
        old_rng, new_rng = np.random.default_rng(404), np.random.default_rng(404)
        expected = [int(old_rng.integers(len(q))) if old_rng.random() < epsilon
                    else int(np.argmax(q)) for q in scores]
        assert infer.choose(params, requests, new_rng, epsilon) == expected
        assert old_rng.bit_generator.state == new_rng.bit_generator.state
    assert infer.choose(params, []) == []
    tied = dict(params, **{'qout.w': np.zeros_like(params['qout.w']),
                         'qout.b': np.zeros_like(params['qout.b'])})
    assert infer.choose(tied, requests) == [0]*5
    bad = dict(params, **{'qout.b': np.full(3, np.nan, np.float32)})
    with pytest.raises(FloatingPointError):
        infer.choose(bad, requests[1:2])


def test_eval_buckets_preserve_choices(policy):
    from dougpu.inference import Inference
    from dougpu.reference import ReferenceGame
    mc, params = policy
    requests = [encode(ReferenceGame(np.random.default_rng(i)).public()) for i in range(35)]
    fixed = Inference(mc, batch=64, action_chunk=2048)
    bucketed = Inference(mc, batch=64, action_chunk=2048, bucket_batch=True)
    for n in (1, 8, 9, 17, 35):
        assert fixed.choose(params, requests[:n]) == bucketed.choose(params, requests[:n])


@pytest.mark.parametrize('engine', ['reference', 'douzero'])
@pytest.mark.parametrize('workers', [1, 2])
def test_parallel_evaluation_preserves_order_and_training_games(engine, workers):
    if engine == 'douzero':
        pytest.importorskip('douzero.env.game')
    from dougpu.evaluation import paired_evaluate
    class FirstPolicy:
        def choose(self, p, requests):
            return [int(np.argmax(r['actions'][:, :15].sum(axis=1))) for r in requests]
    cfg = TrainConfig(engine=engine, workers=2, envs_per_worker=2)
    pool = ActorPool(cfg, [91, 92])
    requests = [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in r.items()}
                for r in pool.requests]
    try:
        for opponent in ('rule', 'random', {}):
            expected = paired_evaluate({}, opponent, FirstPolicy(), engine, 5, 44)
            actual = paired_evaluate({}, opponent, FirstPolicy(), engine, 5, 44,
                                     actor_pool=pool, workers=workers)
            assert expected == actual
        assert paired_evaluate({}, 'rule', FirstPolicy(), engine, 5, 44,
                               lambda: True, actor_pool=pool, workers=workers) is None
        for a, b in zip(requests, pool.requests):
            same_request(a, b)
        rings = [Ring(engine, 2, s) for s in (91, 92)]
        expected = []
        for ring in rings:
            ring.requests()
            ring.advance([0, 0], 3)
            expected.extend(ring.requests())
        pool.step([0, 0, 0, 0], 3)
        for a, b in zip(expected, pool.requests):
            same_request(a, b)
    finally:
        pool.close()


def test_cpu_profile_respects_parent_limits(tmp_path):
    from dougpu.resources import cpu_profile
    root = tmp_path/'cgroup'
    child = root/'parent'/'child'
    child.mkdir(parents=True)
    (root/'cpu.max').write_text('max 100000')
    (child/'cpu.max').write_text('max 100000')
    (child.parent/'cpu.max').write_text('350000 100000')
    membership = tmp_path/'membership'
    membership.write_text('0::/parent/child\n')
    result = cpu_profile(root, membership, affinity=44)
    assert result['cpu_budget'] == 3.5 and result['workers'] == 2
    assert result['infer_batch'] == 64 and result['envs_per_worker'] == 32
    (child.parent/'cpu.max').write_text('max 100000')
    result = cpu_profile(root, membership, affinity=44)
    assert result['workers'] == 32 and result['infer_batch'] == 256
