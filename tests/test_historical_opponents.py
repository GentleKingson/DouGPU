from dataclasses import asdict, replace
import numpy as np
import pytest

from dougpu.actors import Ring, ActorPool
from dougpu.config import ModelConfig, TrainConfig
from dougpu.checkpoint import policy_bytes, sha256_file
from dougpu.encoding import pack_requests, unpack_requests, PackedRequests, terminal_target
from dougpu.opponents import load_opponents, choose_with_opponents, policy_ids
from dougpu.semantics import expected_parameter_shapes, check_training_semantics


@pytest.mark.parametrize('engine', ['reference', 'douzero'])
def test_frozen_assignment_and_only_learner_trajectories(engine, monkeypatch):
    if engine == 'douzero':
        pytest.importorskip('douzero.env.game')
    import dougpu.actors as actors
    original_label = actors.oracle_label
    calls = []
    def label(hands, role):
        calls.append(role)
        return original_label(hands, role)
    monkeypatch.setattr(actors, 'oracle_label', label)
    ring = Ring(engine, 8, 81, opponents=3, historical_fraction=.5)
    expected, completed, used = [[] for _ in ring.games], 0, set()
    for step in range(240):
        games, matches = list(ring.games), list(ring.matches)
        req = ring.requests()
        ids = [r['policy_id'] for r in req]
        used.update(ids)
        packed = PackedRequests.merge([pack_requests(req[:3]), pack_requests(req[3:])])
        assert policy_ids(packed).tolist() == ids
        assert policy_ids(packed[2:6]).tolist() == ids[2:6]
        for a, b in zip(req, unpack_requests(packed.data)):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        choices = [int(np.argmax(r['actions'][:, :15].sum(axis=1))) for r in req]
        for i, r in enumerate(req):
            opponent, own_landlord = matches[i]
            assert r['policy_id'] == (0 if (r['role'] == 0) == own_landlord else opponent)
            if not r['policy_id']:
                expected[i].append((r['tokens'], r['state'], r['actions'][choices[i]], r['role'],
                                    original_label(games[i].oracle(), r['role']), step//9))
        calls.clear()
        actual, winners = ring.advance(choices, step//9)
        assert len(calls) == ids.count(0)
        target_rows, target_winners = [], []
        for i, game in enumerate(games):
            if game is ring.games[i]:
                assert ring.matches[i] == matches[i]
            else:
                completed += 1
                target_winners.append(game.winner)
                for t, s, a, role, belief, version in expected[i]:
                    target_rows.append((t, s, a, role, belief, terminal_target(game.winner, role), version))
                expected[i] = []
        assert len(actual) == len(target_rows) and winners == target_winners
        for a, b in zip(actual, target_rows):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
    assert completed > 20 and used == {0, 1, 2, 3}


@pytest.mark.parametrize('packed', [False, True])
def test_historical_actor_protocol_matches_local_ring(packed):
    cfg = TrainConfig(engine='reference', workers=2, envs_per_worker=2,
                      historical_opponents=['a.npz', 'b.npz', 'c.npz'])
    rings = [Ring('reference', 2, seed, 3, .5) for seed in (91, 92)]
    pool = ActorPool(cfg, [91, 92], packed=packed)
    try:
        for step in range(80):
            req = [r for ring in rings for r in ring.requests()]
            for a, b in zip(req, pool.requests):
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
            choices = [int(np.argmax(r['actions'][:, :15].sum(axis=1))) for r in req]
            rows, wins = [], []
            for i, ring in enumerate(rings):
                r, w = ring.advance(choices[i*2:i*2+2], step)
                rows.extend(r)
                wins.extend(w)
            actual, actual_wins = pool.step(choices, step)
            from dougpu.replay import PackedSamples
            expected = PackedSamples.pack(rows)
            assert actual_wins == wins and set(actual.data) == set(expected.data)
            for key in expected.data:
                np.testing.assert_array_equal(actual.data[key], expected.data[key])
    finally:
        pool.close()


def test_policy_dispatch_freezes_opponents_and_explores_only_learner():
    req = Ring('reference', 5, 11).requests()
    for r, index in zip(req, [2, 0, 1, 0, 2]):
        r['policy_id'] = index
        r['actions'] = np.tile(r['actions'], (3, 1))
    class Policy:
        def __init__(self):
            self.calls = []
        def choose(self, p, requests, rng=None, epsilon=0):
            self.calls.append((p, [r['policy_id'] for r in requests], rng, epsilon))
            self.last_stats = {'requests': len(requests)}
            if rng is not None:
                for _ in requests:
                    rng.random()
            return [p]*len(requests)
    for batch in (req, PackedRequests.merge([pack_requests(req)])):
        infer = Policy()
        rng, expected_rng = np.random.default_rng(44), np.random.default_rng(44)
        assert choose_with_opponents(infer, 0, [1, 2], batch, rng, .1) == [2, 0, 1, 0, 2]
        expected_rng.random(2)
        assert rng.bit_generator.state == expected_rng.bit_generator.state
        assert infer.last_stats['requests'] == 5
        assert infer.calls[0] == (0, [0, 0], rng, .1)
        assert infer.calls[1:] == [(1, [1], None, 0.), (2, [2, 2], None, 0.)]
    req[0]['policy_id'] = 3
    with pytest.raises(ValueError, match='Unknown'):
        choose_with_opponents(Policy(), 0, [1, 2], req, np.random.default_rng(0), .1)


def test_historical_policy_loading_validates_model_arrays_and_hash(tmp_path):
    model = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16)
    params = {k: np.zeros(shape, np.float32) for k, shape in expected_parameter_shapes(model).items()}
    file = tmp_path/'policy.npz'
    file.write_bytes(policy_bytes(params, asdict(model)))
    policies, hashes = load_opponents([str(file)], model)
    assert hashes == [sha256_file(file)] and 'ntp' not in policies[0]
    with pytest.raises(ValueError, match='model mismatch'):
        load_opponents([str(file)], replace(model, width=32))
    params['qout.w'][0, 0] = np.nan
    file.write_bytes(policy_bytes(params, asdict(model)))
    with pytest.raises(ValueError, match='array'):
        load_opponents([str(file)], model)


def test_historical_algorithm_cannot_change_during_hardware_tuning():
    base = TrainConfig()
    historical = replace(base, historical_opponents=['a.npz'])
    for old, new in ((base, historical), (historical, base),
                     (historical, replace(historical, historical_fraction=.75))):
        with pytest.raises(ValueError, match='semantics changed'):
            check_training_semantics(asdict(old), new)
    check_training_semantics(asdict(historical), replace(historical, workers=8))
    for value in (None, [''], ['same', 'same']):
        with pytest.raises(ValueError, match='historical_opponents'):
            replace(base, historical_opponents=value).validate()
    for value in (0, 1.1, float('nan')):
        with pytest.raises(ValueError, match='historical_fraction'):
            replace(base, historical_fraction=value).validate()
    with pytest.raises(ValueError, match='ordered actors'):
        replace(historical, selfplay_kv_cache=True).validate()


def test_resume_rejects_changed_frozen_bytes_at_same_path(tmp_path, monkeypatch):
    import json
    import jax
    from dougpu.train import main
    model = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16)
    params = {k: np.zeros(s, np.float32) for k, s in expected_parameter_shapes(model).items()}
    policy = tmp_path/'opponent.npz'
    policy.write_bytes(policy_bytes(params, asdict(model)))
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'model': asdict(model), 'train': {
        'backend': 'cuda' if jax.default_backend() == 'gpu' else jax.default_backend(),
        'require_tpu': False, 'engine': 'reference', 'workers': 1, 'envs_per_worker': 1,
        'max_hours': 1e-9, 'historical_opponents': [str(policy)]}}))
    monkeypatch.setattr('sys.argv', ['train', '--config', str(config), '--workdir', str(tmp_path/'run')])
    main()
    params['qout.b'][:] = .125
    policy.write_bytes(policy_bytes(params, asdict(model)))
    with pytest.raises(ValueError, match='bytes/order changed'):
        main()
