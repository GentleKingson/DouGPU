import numpy as np
import pytest

from dougpu.config import TrainConfig
from dougpu.evaluation import _summarize, paired_difference, promotion_decision, validate_holdout_seed
from dougpu.checkpoint import Store, load_policy, policy_bytes


def summary(landlord, farmer, n=256):
    outcomes = np.zeros((n, 2), np.float32)
    outcomes[:round(n*landlord), 0] = 1
    outcomes[:round(n*farmer), 1] = 1
    return _summarize(outcomes, 900001)


def test_role_intervals_and_paired_difference_retain_deal_unit():
    result = _summarize(np.tile([[1, 0], [0, 1]], (128, 1)), 123)
    assert result['balanced_bootstrap_ci95'] == [.5, .5]
    assert result['paired_bootstrap_ci95'] == [.5, .5]
    assert result['landlord_bootstrap_ci95'][0] < .5 < result['landlord_bootstrap_ci95'][1]
    assert result['farmer_team_bootstrap_ci95'][0] < .5 < result['farmer_team_bootstrap_ci95'][1]
    same = paired_difference(result, result)
    for role in ('landlord', 'farmer_team', 'balanced'):
        assert same[role+'_win_rate'] == 0
        assert same[role+'_bootstrap_ci95'] == [0, 0]
    all_wins, all_losses = summary(1, 1), summary(0, 0)
    gain = paired_difference(all_wins, all_losses)
    assert gain['balanced_bootstrap_ci95'] == [1, 1]
    with pytest.raises(ValueError, match='identical'):
        paired_difference(result, dict(result, seed=124))
    with pytest.raises(ValueError, match='Invalid'):
        paired_difference(result, dict(result, paired_outcomes=[[1, 1]]))


def test_promotion_blocks_role_regression_despite_balanced_gain():
    no_rule_change = paired_difference(summary(.6, .6), summary(.6, .6))
    head = summary(.95, .4)
    assert head['paired_bootstrap_ci95'][0] > .5  # old gate would promote
    decision = promotion_decision(head, no_rule_change, .05)
    assert not decision['promoted']
    assert not decision['checks']['farmer_team_head_to_head']
    strong_head = summary(.8, .8)
    rule_regression = paired_difference(summary(.8, .5), summary(.5, .8))
    decision = promotion_decision(strong_head, rule_regression, .05)
    assert not decision['promoted']
    assert not decision['checks']['farmer_team_rule_noninferiority']
    assert promotion_decision(strong_head, no_rule_change, .05)['promoted']
    head_small = summary(1, 1, n=32)
    small_difference = paired_difference(summary(1, 1, n=32), summary(0, 0, n=32))
    assert not promotion_decision(head_small, small_difference, .05)['promoted']
    with pytest.raises(ValueError, match='matching'):
        promotion_decision(strong_head, dict(no_rule_change, seed=123), .05)


@pytest.mark.parametrize('margin', [-.01, .5, float('nan'), float('inf')])
def test_invalid_margin_rejected(margin):
    with pytest.raises(ValueError, match='promotion_role_margin'):
        TrainConfig(promotion_role_margin=margin).validate()
    with pytest.raises(ValueError, match='margin'):
        promotion_decision({}, {}, margin)


def test_exports_remember_selection_seeds(tmp_path):
    params = {'test': np.ones(1, np.float32)}
    opt = {'step': np.array(0), 'm': params, 'v': params}
    meta = {'model': {}, 'cycle': 1, 'updates': 0, 'champion_cycle': 0,
            'selection_seeds': [900002], 'train': {'eval_seed': 900001}}
    Store(tmp_path).save(params, opt, params, None, meta)
    for name in ('best', 'latest'):
        _, policy = load_policy(tmp_path/(name+'_policy.npz'))
        assert policy['metadata']['selection_seeds'] == [900001, 900002]
        with pytest.raises(ValueError, match='held-out'):
            validate_holdout_seed(900002, policy['metadata']['selection_seeds'])
        validate_holdout_seed(910001, policy['metadata']['selection_seeds'])


def test_evaluate_defaults_to_champion_and_rejects_selection_seed(tmp_path, monkeypatch):
    import run_local
    from dataclasses import asdict
    from dougpu.config import ModelConfig
    model, config = ModelConfig(), TrainConfig(backend='cpu', require_tpu=False)
    spec = {'model': asdict(model), 'train': asdict(config)}
    monkeypatch.setattr(run_local, 'prepared', lambda _: (spec, model, config, {}))
    calls = []
    monkeypatch.setattr(run_local, 'invoke', lambda module, argv: calls.append(argv))
    monkeypatch.setattr('sys.argv', ['run_local.py', 'evaluate', '--run-dir', str(tmp_path)])
    run_local.main()
    assert str(tmp_path/'checkpoints'/'best_policy.npz') in map(str, calls[0])
    monkeypatch.setattr('sys.argv', ['run_local.py', 'evaluate', '--run-dir', str(tmp_path),
                                    '--seed', str(config.eval_seed)])
    with pytest.raises(ValueError, match='held-out'):
        run_local.main()
    assert len(calls) == 1


def test_direct_policy_evaluation_rejects_historical_selection_seed_before_jax(tmp_path, monkeypatch):
    from dougpu.evaluate import main
    policy = tmp_path/'policy.npz'
    policy.write_bytes(policy_bytes({}, {}, {'selection_seeds': [900002]}))
    monkeypatch.setattr('sys.argv', ['evaluate', '--policy', str(policy), '--seed', '900002'])
    with pytest.raises(ValueError, match='held-out'):
        main()
