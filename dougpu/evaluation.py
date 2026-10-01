"""Paired fixed-deal evaluation. Opponent farmers use the same policy.

This is a smoke/iteration metric, not a claim of strength against DouZero weights.
A deal (not an individual game) is the bootstrap sampling unit.
"""
import numpy as np
from .environment import make_game
from .encoding import DECK, encode, pack_requests


def rule_action(public):
    # Deliberately simple and imperfect; never label this a strong baseline.
    actions = public.legal
    hand_size = len(public.hand)
    winning = [i for i, a in enumerate(actions) if len(a) == hand_size]
    if winning:
        return winning[0]
    # Avoid overtaking a farmer teammate if it still owns the current trick.
    passes = [i for i, a in enumerate(actions) if not a]
    if public.role != 0 and passes:
        lead = next(((role, a) for role, a in reversed(public.history) if a), None)
        if lead is not None and lead[0] not in (0, public.role):
            return passes[0]
    nonpass = [(i, a) for i, a in enumerate(actions) if a]
    if not nonpass:
        return passes[0]
    # Shed cards, save high cards. Real strategy is much more subtle.
    return max(nonpass, key=lambda ia: (len(ia[1]) - .025*sum(ia[1]), -max(ia[1])))[0]


class EvaluationGames:
    def __init__(self, engine, jobs, use_rule):
        rng = np.random.default_rng(0)
        self.jobs = [(i, current, make_game(engine, rng, deal)) for i, current, deal in jobs]
        self.use_rule = use_rule

    def observe(self):
        publics = [game.public() for _, _, game in self.jobs]
        current = [(p.role == 0) == seat for p, (_, seat, _) in zip(publics, self.jobs)]
        return ([i for i, _, _ in self.jobs],
                current,
                pack_requests([encode(p) for p, own in zip(publics, current)
                               if not self.use_rule or own]),
                [rule_action(p) if self.use_rule and not own else 0
                 for p, own in zip(publics, current)])

    def advance(self, choices):
        if len(choices) != len(self.jobs):
            raise ValueError('Evaluation request/response size mismatch')
        still, outcomes = [], []
        for job, choice in zip(self.jobs, choices):
            i, current, game = job
            game.step(int(choice))
            if game.done:
                outcomes.append((i, float((game.winner == 0) == current)))
            else:
                still.append(job)
        self.jobs = still
        return self.observe(), outcomes


def _policy_choices(infer, params, opponent, capacity, kv_cache):
    if not kv_cache:
        return lambda p, requests, keys: infer.choose(p, requests)
    from .kv_inference import KVInference
    current = KVInference(infer.mc, capacity, infer.batch, infer.chunk, bucket_batch=infer.bucket_batch)
    other = None if isinstance(opponent, str) or hasattr(opponent, 'choose_games') else KVInference(
        infer.mc, capacity, infer.batch, infer.chunk, bucket_batch=infer.bucket_batch)

    def choose(p, requests, keys):
        policy = current if p is params else other
        return policy.choose(p, requests, cache_keys=keys, policy_version=0)
    return choose


def _parallel_evaluate(params, opponent, choose, n_deals, seed, stop_requested, actor_pool, workers):
    from .actors import EvaluationPool
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_deals):
        deal = rng.permutation(DECK).tolist()
        jobs.extend([(2*i, True, deal), (2*i+1, False, deal)])
    pool = EvaluationPool(actor_pool, jobs, workers, isinstance(opponent, str) and opponent == 'rule')
    outcomes = np.zeros((n_deals, 2), np.float32)
    try:
        while pool.ids:
            if stop_requested():
                return None
            current = [i for i, value in enumerate(pool.current) if value]
            other = [i for i, value in enumerate(pool.current) if not value]
            choices = np.zeros(len(pool.ids), np.int32)
            if current:
                choices[current] = choose(params, [pool.requests[i] for i in current],
                                          [pool.ids[i] for i in current])
            if other:
                if isinstance(opponent, str):
                    if opponent == 'rule':
                        choices[other] = [pool.rules[i] for i in other]
                    elif opponent == 'random':
                        choices[other] = [rng.integers(len(pool.requests[i]['actions'])) for i in other]
                    else:
                        raise ValueError('Unknown evaluation opponent')
                else:
                    choices[other] = choose(opponent, [pool.requests[i] for i in other],
                                           [pool.ids[i] for i in other])
            for i, win in pool.step(choices):
                outcomes[i//2, i % 2] = win
    finally:
        pool.close()
    return _summarize(outcomes, seed)


def paired_evaluate(params, opponent, infer, engine, n_deals=32, seed=900001,
                    stop_requested=lambda: False, *, actor_pool=None, workers=0, kv_cache=False):
    if n_deals < 1:
        raise ValueError('Evaluation needs at least one paired deal')
    choose = _policy_choices(infer, params, opponent, 2*n_deals, kv_cache)
    if workers and actor_pool is not None and not hasattr(opponent, 'choose_games'):
        if actor_pool.cfg.engine != engine:
            raise ValueError('Evaluation and training rule engines must match')
        return _parallel_evaluate(params, opponent, choose, n_deals, seed, stop_requested, actor_pool, workers)
    # Same immutable deal for current-landlord and current-farmer assignments.
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_deals):
        deal = rng.permutation([r for r in list(range(3, 15))+[17] for _ in range(4)] + [20, 30]).tolist()
        for current_landlord in (True, False):
            jobs.append((i, current_landlord, make_game(engine, rng, deal)))
    outcomes = np.zeros((n_deals, 2), np.float32)
    pending = list(range(len(jobs)))
    while pending:
        if stop_requested():
            return None
        publics = [jobs[j][2].public() for j in pending]
        current = [i for i, (j, pub) in enumerate(zip(pending, publics))
                   if (pub.role == 0) == jobs[j][1]]
        current_set = set(current)
        other = [i for i in range(len(pending)) if i not in current_set]
        choices = np.zeros(len(pending), np.int32)
        if current:
            choices[current] = choose(params, [encode(publics[i]) for i in current],
                                      [pending[i] for i in current])
        if other:
            if hasattr(opponent, 'choose_games'):
                selected = opponent.choose_games([jobs[pending[i]][2] for i in other])
                for i, choice in zip(other, selected):
                    choices[i] = choice
            elif isinstance(opponent, str):
                if opponent == 'random':
                    for i in other:
                        choices[i] = int(rng.integers(len(publics[i].legal)))
                elif opponent == 'rule':
                    for i in other:
                        choices[i] = rule_action(publics[i])
                else:
                    raise ValueError('Unknown evaluation opponent')
            else:
                choices[other] = choose(opponent, [encode(publics[i]) for i in other],
                                       [pending[i] for i in other])
        still = []
        for j, choice in zip(pending, choices):
            deal_id, current_l, game = jobs[j]
            game.step(int(choice))
            if game.done:
                win = (game.winner == 0) == current_l
                outcomes[deal_id, 0 if current_l else 1] = float(win)
            else:
                still.append(j)
        pending = still
    return _summarize(outcomes, seed)


def _summarize(outcomes, seed):
    outcomes = np.asarray(outcomes)
    if (outcomes.ndim != 2 or outcomes.shape[1] != 2 or not len(outcomes)
            or not np.isin(outcomes, [0, 1]).all()):
        raise ValueError('Expected nonempty binary paired outcomes')
    return dict(_summarize_values(outcomes, seed), paired_outcomes=outcomes.astype(int).tolist())


def _summarize_values(outcomes, seed):
    n_deals = len(outcomes)
    # Resample deals to retain dependence between the two policy assignments.
    bootstrap_rng = np.random.default_rng(seed+12345)
    draws = bootstrap_rng.integers(n_deals, size=(2000, n_deals))
    values = np.column_stack((outcomes, outcomes.mean(axis=1)))
    means = values[draws].mean(axis=1)
    intervals = np.quantile(means, [.025, .975], axis=0)
    names = ('landlord', 'farmer_team', 'balanced')
    result = {'deals': n_deals, 'games': 2*n_deals, 'seed': seed,
              # Five promotion checks: one-sided Bonferroni bounds, per look only.
              'promotion_lower_bounds': dict(zip(names, np.quantile(means, .01, axis=0).tolist()))}
    for i, name in enumerate(names):
        result[name+'_win_rate'] = float(values[:, i].mean())
        result[name+'_bootstrap_ci95'] = intervals[:, i].tolist()
    result['paired_bootstrap_ci95'] = result['balanced_bootstrap_ci95']
    return result


def paired_difference(candidate, reference):
    """Same opponent, engine and ordered deals are required by the caller."""
    if (candidate['seed'], candidate['deals']) != (reference['seed'], reference['deals']):
        raise ValueError('Paired differences require identical evaluation deals')
    a, b = (np.asarray(result['paired_outcomes']) for result in (candidate, reference))
    if (a.shape != (candidate['deals'], 2) or b.shape != a.shape
            or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all()):
        raise ValueError('Invalid paired outcomes')
    return dict(_summarize_values(a-b, candidate['seed']), metric='candidate_minus_reference')


def promotion_decision(head_to_head, rule_difference, role_margin):
    """Balanced superiority plus role noninferiority, including a fixed reference."""
    if not np.isfinite(role_margin) or not 0 <= role_margin < .5:
        raise ValueError('Invalid promotion role margin')
    if ((head_to_head['seed'], head_to_head['deals']) != (rule_difference['seed'], rule_difference['deals'])
            or rule_difference.get('metric') != 'candidate_minus_reference'):
        raise ValueError('Promotion requires matching paired rule differences')
    checks = {'enough_deals': head_to_head['deals'] >= 128,
              'balanced_superiority': head_to_head['promotion_lower_bounds']['balanced'] > .5}
    for role in ('landlord', 'farmer_team'):
        checks[role+'_head_to_head'] = head_to_head['promotion_lower_bounds'][role] >= .5-role_margin
        checks[role+'_rule_noninferiority'] = rule_difference['promotion_lower_bounds'][role] >= -role_margin
    return {'promoted': all(checks.values()), 'checks': checks, 'role_margin': role_margin,
            'bounds': 'one-sided 99% paired bootstrap; Bonferroni for five checks per look'}


def validate_holdout_seed(seed, selection_seeds):
    if seed in selection_seeds:
        raise ValueError('Use held-out seeds, different from champion selection seeds')
