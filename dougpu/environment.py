"""Thin adapter around the upstream CPU rules engine, with no PyTorch imports."""
from functools import lru_cache
from types import SimpleNamespace
from .encoding import DECK, POSITIONS, MAX_SEQ, PublicState, tokenize, move_tokens


@lru_cache(maxsize=2)
def _environment_type(compact):
    try:
        from douzero.env.game import GameEnv
    except ImportError as exc:
        raise RuntimeError("DouZero environment missing. Run bootstrap.py first; no silent fallback.") from exc
    if not compact:
        return GameEnv

    class CompactGameEnv(GameEnv):
        def get_infoset(self):
            # Rules/step remain upstream. Only the unused observation deepcopy is removed.
            role = self.acting_player_position
            return SimpleNamespace(player_position=role,
                player_hand_cards=tuple(self.info_sets[role].player_hand_cards),
                num_cards_left_dict={p: len(self.info_sets[p].player_hand_cards) for p in POSITIONS},
                legal_actions=self.get_legal_card_play_actions())
    return CompactGameEnv

class _ActionAgent:
    def __init__(self):
        self.action = None
    def act(self, infoset):
        return self.action


class DouZeroGame:
    def __init__(self, rng, deal=None, *, compact=True):
        deck = list(map(int, rng.permutation(DECK))) if deal is None else list(map(int, deal))
        self.bottom = tuple(sorted(deck[17:20]))
        self.players = {p: _ActionAgent() for p in POSITIONS}
        self.game = _environment_type(compact)(self.players)
        self.game.card_play_init({POSITIONS[0]: sorted(deck[:20]),
                                  POSITIONS[1]: sorted(deck[20:37]),
                                  POSITIONS[2]: sorted(deck[37:]),
                                  "three_landlord_cards": list(self.bottom)})
        self.history = []
        self._tokens = None
        self.done, self.winner = False, None
        self._refresh()
        self._tokens = tokenize(self.public()).tobytes()

    def _refresh(self):
        info = self.game.game_infoset
        self.role = POSITIONS.index(info.player_position)
        # Deduplicate physically identical actions. Preserve all distinct legal moves.
        self.legal = tuple(sorted(set(tuple(sorted(a)) for a in info.legal_actions)))

    def public(self):
        i = self.game.game_infoset
        return PublicState(self.role, tuple(i.player_hand_cards),
            tuple(i.num_cards_left_dict[p] for p in POSITIONS), self.bottom,
            tuple(self.history), self.legal, self._tokens)

    def full_infoset(self):
        # Official DouZero opponents need fields that the training policy never consumes.
        return _environment_type(False).get_infoset(self.game)

    def oracle(self):
        return [self.game.info_sets[p].player_hand_cards for p in POSITIONS]

    def step(self, index):
        if self.done or not 0 <= index < len(self.legal):
            raise ValueError("Invalid step")
        move = self.legal[index]
        self.players[POSITIONS[self.role]].action = list(move)
        self.history.append((self.role, move))
        self._tokens += move_tokens(self.role, move)
        if len(self._tokens) > MAX_SEQ:
            raise RuntimeError("History overflow: do not silently discard public information")
        self.game.step()
        self.done = self.game.game_over
        if self.done:
            self.winner = self.role
        else:
            if len(self.history) > 162:
                raise RuntimeError("Rules invariant broken: excessive game length")
            self._refresh()


def make_game(engine, rng, deal=None):
    if engine == "douzero":
        return DouZeroGame(rng, deal)
    if engine == "reference":
        from .reference import ReferenceGame
        return ReferenceGame(rng, deal)
    raise ValueError(engine)
