"""Compact NumPy ring; only completed episodes are inserted by actors."""
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
import time
from .encoding import MAX_SEQ, STATE_DIM, ACTION_DIM, BELIEF_DIM, merge_token_batches


def group_history(batch, groups):
    """Reorder the already sampled rows; retain their global role weights."""
    accumulation, micro = batch['target'].shape
    if groups < 1 or accumulation % groups:
        raise ValueError('History groups must divide accumulation')
    count = accumulation * micro
    flat = {key: value.reshape((count,) + value.shape[2:]) for key, value in batch.items()}
    order = np.argsort(flat['lengths'], kind='stable')
    output = []
    for selected in np.split(order, groups):
        length = next(n for n in (64, 128, 256, 512) if n >= int(flat['lengths'][selected].max()))
        # Slice before advanced indexing so PAD columns are never copied.
        values = {key: (value[:, :length] if key == 'tokens' else value)[selected]
                  for key, value in flat.items()}
        output.append({key: value.reshape((accumulation//groups, micro) + value.shape[1:])
                       for key, value in values.items()})
    return tuple(output)


def _arrays(capacity, token_width=MAX_SEQ):
    return {"tokens": np.zeros((capacity, token_width), np.uint8),
                     "lengths": np.zeros(capacity, np.int32),
                     "state": np.zeros((capacity, STATE_DIM), np.float16),
                     "actions": np.zeros((capacity, ACTION_DIM), np.float16),
                     "role": np.zeros(capacity, np.int32),
                     "belief": np.zeros((capacity, BELIEF_DIM), np.float16),
                     "target": np.zeros(capacity, np.float32),
                     "version": np.zeros(capacity, np.int32)}


@dataclass
class PackedSamples:
    data: dict

    def __len__(self):
        return len(self.data.get('lengths', ()))

    @classmethod
    def pack(cls, rows):
        if not rows:
            return cls({})
        tokens, state, actions, role, belief, target, version = zip(*rows)
        lengths = np.asarray([len(t) for t in tokens], np.int32)
        if np.any(lengths < 1) or np.any(lengths > MAX_SEQ):
            raise ValueError('Invalid sample history length')
        out = _arrays(len(rows), int(lengths.max()))
        for i, t in enumerate(tokens):
            out['tokens'][i, :lengths[i]] = t
        for key, value in (('lengths', lengths), ('state', state), ('actions', actions),
                           ('role', role), ('belief', belief), ('target', target), ('version', version)):
            out[key][:] = value
        return cls(out)

    @classmethod
    def merge(cls, batches):
        batches = [b for b in batches if len(b)]
        if not batches:
            return cls({})
        if len(batches) == 1:
            return batches[0]
        return cls(merge_token_batches([b.data for b in batches]))


class ReplayPrefetch:
    """One batch ahead, with RNG committed only when the learner consumes it."""
    def __init__(self, replay, cfg, rng, cycle, device_put, strata=None):
        self.replay, self.cfg, self.rng, self.cycle = replay, cfg, rng, cycle
        self.strata = strata
        self.local_rng = deepcopy(rng)
        self.device_put = device_put
        self.pending = None
        self.sample_seconds = 0.

    def prepare(self):
        if self.pending is not None:
            raise RuntimeError('Only one prefetched batch is allowed')
        tick = time.monotonic()
        try:
            batch = self.replay.sample(self.cfg, self.local_rng, self.cycle, self.strata)
            length = batch['tokens'].shape[-1]
            state = self.local_rng.bit_generator.state
            self.pending = (self.device_put(batch), length, state)
        except Exception as exc:
            # Surface preparation failures only after the already-dispatched
            # optimizer step has been synchronized and accounted for.
            self.pending = exc
        finally:
            self.sample_seconds += time.monotonic()-tick

    def take(self):
        if self.pending is None:
            raise RuntimeError('Prepare a batch before taking it')
        if isinstance(self.pending, Exception):
            raise self.pending
        batch, length, state = self.pending
        self.pending = None
        self.rng.bit_generator.state = state
        return batch, length


class Replay:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError('Replay capacity must be positive')
        self.capacity, self.pos, self.size = capacity, 0, 0
        self.data = _arrays(capacity)

    def add(self, samples):
        batch = samples if isinstance(samples, PackedSamples) else PackedSamples.pack(samples)
        n = len(batch)
        if not n:
            return
        if set(batch.data) != set(self.data) or any(
                v.shape != (n,) + self.data[k].shape[1:]
                for k, v in batch.data.items() if k != 'tokens'):
            raise ValueError('Invalid packed sample shape')
        tokens = batch.data['tokens']
        if tokens.ndim != 2 or tokens.shape[0] != n or not 1 <= tokens.shape[1] <= MAX_SEQ:
            raise ValueError('Invalid packed token shape')
        if np.any(batch.data['lengths'] < 1) or np.any(batch.data['lengths'] > tokens.shape[1]):
            raise ValueError('Invalid sample history length')
        # Keep the same physical ring indices even when a batch exceeds capacity.
        skip = max(0, n - self.capacity)
        start = (self.pos + skip) % self.capacity
        count = n - skip
        first = min(count, self.capacity - start)
        for k, dst in self.data.items():
            src = batch.data[k][skip:]
            if k == 'tokens':
                # Clear stale suffixes when a shorter history overwrites a ring slot.
                width = src.shape[1]
                dst[start:start+first, width:] = 0
                dst[:count-first, width:] = 0
                dst = dst[:, :width]
            dst[start:start+first] = src[:first]
            dst[:count-first] = src[first:]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def eligible(self, cycle, max_age):
        ids = np.arange(self.size)
        return ids[self.data["version"][:self.size] >= cycle-max_age]

    def strata(self, cfg, cycle):
        """Physical indices, reusable only while replay and sampling version stay fixed."""
        age = cfg.replay_max_age * (cfg.updates_per_cycle if cfg.replay_version_updates else 1)
        ids = self.eligible(cycle, age)
        roles = self.data['role'][ids]
        return tuple(ids[roles == role] for role in range(3))

    def sample(self, cfg, rng, cycle, strata=None):
        bins = self.strata(cfg, cycle) if strata is None else strata
        if not any(len(b) for b in bins):
            raise RuntimeError("No fresh enough replay data")
        # Equal-sized strata up to rounding; shuffle so microbatches mix roles.
        if any(not len(b) for b in bins):
            raise RuntimeError("All three role strata must be present before learning")
        selected = []
        for role, bucket in enumerate(bins):
            n = cfg.batch_size//3 + (role < cfg.batch_size % 3)
            selected.extend(rng.choice(bucket, n, replace=True))
        selected = np.asarray(selected, np.int32)
        rng.shuffle(selected)
        longest = int(self.data['lengths'][selected].max())
        length = next(n for n in (64, 128, 256, 512) if n >= longest)
        b = {k: v[selected].astype(np.int32 if k in ("lengths", "role") else np.float32)
             for k, v in self.data.items() if k not in ("version", "tokens")}
        b['tokens'] = self.data['tokens'][selected, :length].astype(np.int32)
        role_counts = np.bincount(b["role"], minlength=3)
        b["weight"] = (cfg.batch_size/(3 * role_counts[b["role"]])).astype(np.float32)
        return {k: v.reshape((cfg.accumulation, cfg.micro_batch) + v.shape[1:]) for k, v in b.items()}

    def export(self):
        # Ring ordering is restored with pos, so physically stored indices are kept.
        out = {k: v[:self.size] for k, v in self.data.items()}
        out["_meta"] = np.asarray([self.capacity, self.pos, self.size], np.int64)
        return out

    def restore(self, arrays):
        cap, pos, size = map(int, arrays["_meta"])
        if cap != self.capacity:
            raise ValueError("Replay capacity differs from the saved run")
        for k in self.data:
            self.data[k][:size] = arrays[k]
        self.pos, self.size = pos, size
