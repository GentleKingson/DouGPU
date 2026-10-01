"""Fast environment checks; run on CPU, independently of the TPU learner."""
import argparse
from collections import Counter
import sys
import numpy as np
from .environment import make_game
from .encoding import DECK, encode, oracle_label, terminal_target


def check_engine(engine='douzero', games=100, seed=123):
    rng = np.random.default_rng(seed)
    max_len = max_legal = moves = 0
    for _ in range(games):
        game = make_game(engine, rng)
        original_bottom = game.bottom
        while not game.done:
            pub = game.public()
            obs = encode(pub)
            assert set(obs) == {'tokens', 'state', 'role', 'actions'}
            assert obs['state'].shape == (21,) and obs['actions'].shape[1] == 16
            assert len(obs['tokens']) <= 512
            assert pub.bottom == original_bottom
            assert len(pub.hand) == pub.remaining[pub.role]
            assert len(pub.legal) > 0
            own = Counter(pub.hand)
            for action in pub.legal:
                assert not Counter(action) - own
            all_cards = list(c for hand in game.oracle() for c in hand)
            all_cards += [c for _, action in pub.history for c in action]
            assert Counter(all_cards) == Counter(DECK)
            label = oracle_label(game.oracle(), pub.role)
            assert label.shape == (30,) and np.all((label >= 0) & (label <= 1))
            max_len = max(max_len, len(obs['tokens']))
            max_legal = max(max_legal, len(pub.legal))
            # Mix random moves and low-card shedding; check both lead/reply paths.
            if rng.random() < .5:
                choice = int(rng.integers(len(pub.legal)))
            else:
                choice = max(range(len(pub.legal)), key=lambda i: len(pub.legal[i]))
            game.step(choice)
            moves += 1
        targets = [terminal_target(game.winner, r) for r in range(3)]
        assert targets[1] == targets[2] == -targets[0]
    result = {'engine': engine, 'games': games, 'moves': moves, 'max_tokens_seen': max_len,
              'max_legal_actions_seen': max_legal, 'jax_loaded': 'jax' in sys.modules}
    print('[ENGINE SELFTEST]', result, flush=True)
    assert 'jax' not in sys.modules, 'Environment workers must not initialize JAX'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=['douzero', 'reference'], default='douzero')
    p.add_argument('--games', type=int, default=100)
    a = p.parse_args()
    check_engine(a.engine, a.games)
