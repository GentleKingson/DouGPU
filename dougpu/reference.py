"""Independent offline development engine, NOT the authoritative training rules.

It covers standard rank-based combinations. Edge-case airplane interpretation may
vary between platforms. Production defaults to the unmodified DouZero GameEnv.
Checkpoints record the engine and cannot silently switch between implementations.
"""
from collections import Counter
from itertools import combinations
from .encoding import DECK, PublicState


def combinations_of_cards(cards, n):
    return sorted(set(combinations(sorted(cards), n)))


def enumerate_moves(hand):
    c = Counter(hand)
    moves = {}
    def add(a, kind, key, size=1):
        moves[tuple(sorted(a))] = (kind, key, size)
    for r in sorted(c):
        for n, kind in ((1, 1), (2, 2), (3, 3), (4, 4)):
            if c[r] >= n:
                add([r] * n, kind, r)
        if c[r] >= 3:
            for s in sorted(c):
                if r != s:
                    add([r] * 3 + [s], 6, r)
                    if c[s] >= 2:
                        add([r] * 3 + [s] * 2, 7, r)
        if c[r] == 4:
            rem = [x for x in hand if x != r]
            for a in combinations_of_cards(rem, 2):
                add([r] * 4 + list(a), 13, r)
            for pair in combinations([s for s in sorted(c) if s != r and c[s] >= 2], 2):
                add([r] * 4 + list(pair) * 2, 14, r)
    if c[20] and c[30]:
        add([20, 30], 5, 100)
    for repeat, min_len, kind in ((1, 5, 8), (2, 3, 9), (3, 2, 10)):
        for start in range(3, 15):
            run = []
            for end in range(start, 15):
                if c[end] < repeat:
                    break
                run.append(end)
                if len(run) < min_len:
                    continue
                body = sorted(run * repeat)
                add(body, kind, start, len(run))
                if repeat == 3:
                    rem = [x for x in hand if x not in run]
                    for wings in combinations_of_cards(rem, len(run)):
                        add(body + list(wings), 11, start, len(run))
                    pair_ranks = [s for s in sorted(c) if s not in run and c[s] >= 2]
                    for wings in combinations(pair_ranks, len(run)):
                        add(body + list(wings) * 2, 12, start, len(run))
    return moves


def beats(a, b):
    ka, ra, na = a
    kb, rb, nb = b
    return ka == 5 and kb != 5 or (ka == 4 and kb not in (4, 5)) or (
        ka == kb and na == nb and ra > rb)


class ReferenceGame:
    def __init__(self, rng, deal=None):
        deck = list(rng.permutation(DECK)) if deal is None else list(deal)
        self.hands = [sorted(deck[:20]), sorted(deck[20:37]), sorted(deck[37:])]
        self.bottom = tuple(sorted(deck[17:20]))
        self.history, self.role, self.done, self.winner = [], 0, False, None
        self.lead, self.passes = None, 0
        self._refresh()

    def _refresh(self):
        self.moves = enumerate_moves(self.hands[self.role])
        legal = [a for a, desc in self.moves.items() if self.lead is None or beats(desc, self.lead)]
        if self.lead is not None:
            legal.append(())
        self.legal = tuple(sorted(legal))
        assert self.legal

    def public(self):
        return PublicState(self.role, tuple(self.hands[self.role]), tuple(map(len, self.hands)),
                           self.bottom, tuple(self.history), self.legal)

    def oracle(self):
        return self.hands

    def step(self, index):
        if self.done or not 0 <= index < len(self.legal):
            raise ValueError("Invalid step")
        move = self.legal[index]
        self.history.append((self.role, move))
        if move:
            self.lead, self.passes = self.moves[move], 0
            for r in move:
                self.hands[self.role].remove(r)
            if not self.hands[self.role]:
                self.done, self.winner = True, self.role
                return
        else:
            self.passes += 1
            if self.passes == 2:
                self.lead, self.passes = None, 0
        if len(self.history) > 162:
            raise RuntimeError("Rules invariant broken: excessive game length")
        self.role = (self.role + 1) % 3
        self._refresh()
