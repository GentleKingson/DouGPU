import numpy as np
import pytest

from dougpu.encoding import DECK, unpack_requests
from dougpu.evaluation import EvaluationGames


@pytest.mark.parametrize('use_rule', [False, True])
@pytest.mark.parametrize('assignments', [[], [False], [True], [True, False, True]])
def test_observe_only_computes_consumed_inputs(monkeypatch, use_rule, assignments):
    import dougpu.evaluation as evaluation
    encode, rule = evaluation.encode, evaluation.rule_action
    calls = {'encode': 0, 'rule': 0}

    def counted_encode(public):
        calls['encode'] += 1
        return encode(public)

    def counted_rule(public):
        calls['rule'] += 1
        return rule(public)

    monkeypatch.setattr(evaluation, 'encode', counted_encode)
    monkeypatch.setattr(evaluation, 'rule_action', counted_rule)
    rng = np.random.default_rng(81)
    jobs = [(i, own, rng.permutation(DECK).tolist()) for i, own in enumerate(assignments)]
    games = EvaluationGames('reference', jobs, use_rule)
    while True:
        before = dict(calls)
        ids, current, packed, rules = games.observe()
        req = unpack_requests(packed)
        publics = [game.public() for _, _, game in games.jobs]
        assert ids == [i for i, _, _ in games.jobs]
        assert current == [(p.role == 0) == own for p, (_, own, _) in zip(publics, games.jobs)]
        selected = [i for i, own in enumerate(current) if not use_rule or own]
        assert calls['encode']-before['encode'] == len(selected)
        assert calls['rule']-before['rule'] == (len(ids)-sum(current) if use_rule else 0)
        for i, actual in zip(selected, req):
            expected = encode(publics[i])
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
        assert len(req) == len(selected)
        assert rules == [rule(p) if use_rule and not own else 0 for p, own in zip(publics, current)]
        if not ids:
            break
        choices = [rule(p) if use_rule and not own else max(range(len(p.legal)), key=lambda k: len(p.legal[k]))
                   for p, own in zip(publics, current)]
        games.advance(choices)
