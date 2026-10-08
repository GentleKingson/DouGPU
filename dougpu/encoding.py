"""Explicit public/private boundary. Inference functions never accept oracle labels."""
from dataclasses import dataclass
from collections import Counter
from functools import lru_cache
import numpy as np

RANKS = tuple(range(3, 15)) + (17, 20, 30)
RANK_INDEX = {r: i for i, r in enumerate(RANKS)}
POSITIONS = ("landlord", "landlord_down", "landlord_up")
DECK = tuple(r for r in RANKS[:13] for _ in range(4)) + (20, 30)
PAD, BOS, BOTTOM, END, PASS = 0, 1, 2, 3, 4
SEAT_BASE, RANK_BASE, MAX_SEQ = 5, 8, 512
STATE_DIM, ACTION_DIM, BELIEF_DIM = 21, 16, 30

@dataclass(frozen=True)
class PublicState:
    role: int
    hand: tuple
    remaining: tuple
    bottom: tuple  # original PUBLIC bottom cards, not the engine's mutable list
    history: tuple  # tuple of (absolute role, tuple of played ranks), including passes
    legal: tuple
    history_tokens: bytes | None = None


def counts(cards):
    out = np.zeros(15, np.float32)
    for r, n in Counter(cards).items():
        if r not in RANK_INDEX or n > (4 if r < 20 else 1):
            raise ValueError(f"Invalid card count {r}: {n}")
        out[RANK_INDEX[r]] = n / 4.0
    return out


def tokenize(s: PublicState):
    if s.history_tokens is not None:
        if not 0 < len(s.history_tokens) <= MAX_SEQ:
            raise ValueError("History overflow: do not silently discard public information")
        return np.frombuffer(s.history_tokens, np.uint8).copy()
    tokens = [BOS, BOTTOM] + [RANK_BASE + RANK_INDEX[r] for r in sorted(s.bottom)] + [END]
    for seat, move in s.history:
        tokens.extend(move_tokens(seat, move))
    # At most 54 non-pass moves and 2 passes between non-pass moves:
    # 6 + 54 cards + 2*54 move delimiters + 3*106 pass tokens = 486 < 512.
    if len(tokens) > MAX_SEQ:
        raise ValueError("History overflow: do not silently discard public information")
    return np.asarray(tokens, np.uint8)


def move_tokens(seat, move):
    return bytes([SEAT_BASE + seat] +
                 ([RANK_BASE + RANK_INDEX[r] for r in sorted(move)] if move else [PASS]) + [END])


def pack_requests(requests):
    if not requests:
        return None
    lengths = np.asarray([len(r['tokens']) for r in requests], np.int32)
    tokens = np.zeros((len(requests), int(lengths.max())), np.uint8)
    for i, r in enumerate(requests):
        tokens[i, :lengths[i]] = r['tokens']
    batch = {'tokens': tokens, 'lengths': lengths,
            'state': np.stack([r['state'] for r in requests]),
            'role': np.asarray([r['role'] for r in requests], np.int32),
            'counts': np.asarray([len(r['actions']) for r in requests], np.int32),
            'actions': np.concatenate([r['actions'] for r in requests])}
    if any('policy_id' in r for r in requests):
        batch['policy_id'] = np.asarray([r['policy_id'] for r in requests], np.int32)
    return batch


def unpack_requests(batch):
    if batch is None:
        return []
    offsets = np.r_[0, np.cumsum(batch['counts'])]
    return [{'tokens': batch['tokens'][i, :n], 'state': batch['state'][i],
             'role': int(batch['role'][i]), 'actions': batch['actions'][offsets[i]:offsets[i+1]],
             **({'policy_id': int(batch['policy_id'][i])} if 'policy_id' in batch else {})}
            for i, n in enumerate(batch['lengths'])]


def merge_token_batches(batches):
    data = {k: np.concatenate([b[k] for b in batches])
            for k in batches[0] if k != 'tokens'}
    tokens = np.zeros((len(data['lengths']), max(b['tokens'].shape[1] for b in batches)), np.uint8)
    offset = 0
    for batch in batches:
        n, width = batch['tokens'].shape
        tokens[offset:offset+n, :width] = batch['tokens']
        offset += n
    return dict(data, tokens=tokens)


@dataclass
class PackedRequests:
    """Ordered actor arrays; contiguous slices stay packed until inference."""
    data: dict

    def __post_init__(self):
        self.offsets = np.r_[0, np.cumsum(self.data['counts'])]

    def __len__(self):
        return len(self.data['lengths'])

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            stop = max(start, stop)
            data = {k: v[start:stop] for k, v in self.data.items() if k != 'actions'}
            data['actions'] = self.data['actions'][self.offsets[start]:self.offsets[stop]]
            return PackedRequests(data)
        i = range(len(self))[key]
        return {'tokens': self.data['tokens'][i, :self.data['lengths'][i]],
                'state': self.data['state'][i], 'role': int(self.data['role'][i]),
                'actions': self.data['actions'][self.offsets[i]:self.offsets[i+1]],
                **({'policy_id': int(self.data['policy_id'][i])} if 'policy_id' in self.data else {})}

    @classmethod
    def merge(cls, batches):
        batches = [b for b in batches if b is not None]
        if not batches:
            return []
        if len(batches) == 1:
            return cls(batches[0])
        return cls(merge_token_batches(batches))


def state_features(s: PublicState):
    role = np.eye(3, dtype=np.float32)[s.role]
    return np.concatenate([_action_counts(tuple(s.hand)), np.asarray(s.remaining, np.float32) / 20, role])


@lru_cache(maxsize=32768)
def _action_counts(cards):
    # Count vectors are pure and immutable, so hands can share this bounded cache.
    value = counts(cards)
    value.flags.writeable = False
    return value


def action_features(actions):
    out = np.zeros((len(actions), ACTION_DIM), np.float32)
    for i, a in enumerate(actions):
        out[i, :15] = _action_counts(tuple(a))
        out[i, 15] = not bool(a)
    return out


def encode(s: PublicState):
    if not s.legal:
        raise ValueError("A live decision must have at least one legal action")
    return {"tokens": tokenize(s), "state": state_features(s),
            "role": s.role, "actions": action_features(s.legal)}


def oracle_label(hands, role):
    # Labels only. Never concatenate this array into state/action/token inputs.
    return np.concatenate([_action_counts(tuple(hands[(role + k) % 3])) for k in (1, 2)])


def terminal_target(winner, role):
    """WP objective: both farmers share the SAME team reward."""
    return np.float32(1 if (winner == 0) == (role == 0) else -1)
