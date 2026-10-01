"""Bucketed encoder and complete legal-action chunks; choose returns indices only."""
import numpy as np
import jax
import jax.numpy as jnp
from .model import encode_state, q_values
from .encoding import STATE_DIM, ACTION_DIM, PackedRequests


def validate_attention_tokens(tokens, lengths):
    """Validate host arrays before the cuDNN length-based attention path.

    Any number of leading batch dimensions is allowed (including learner
    accumulation). This check belongs before device_put/JIT; JAX-internal
    encoders rely on the validated nonempty right-padding invariant.
    """
    tokens, lengths = np.asarray(tokens), np.asarray(lengths)
    if tokens.ndim < 2 or lengths.shape != tokens.shape[:-1]:
        raise ValueError('cuDNN attention token/length shapes do not match')
    if not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError('cuDNN attention requires integer history lengths')
    if np.any(lengths < 1) or np.any(lengths > tokens.shape[-1]):
        raise ValueError('cuDNN attention requires nonempty histories within the token width')
    prefix = np.arange(tokens.shape[-1]) < lengths[..., None]
    if not np.array_equal(tokens != 0, prefix):
        raise ValueError('cuDNN attention requires right padding: no PAD inside a history '
                         'and no non-PAD token after its length')


def merge_inference_stats(rows):
    keys = ('requests', 'legal_actions', 'action_slots', 'history_tokens',
            'history_slots', 'state_slots', 'encoder_batches', 'fused_batches')
    stats = {key: sum(row.get(key, 0) for row in rows) for key in keys}
    stats['max_legal_actions'] = max((row.get('max_legal_actions', 0) for row in rows), default=0)
    for name, used, slots in (('action', 'legal_actions', 'action_slots'),
                              ('history', 'history_tokens', 'history_slots'),
                              ('state', 'requests', 'state_slots')):
        stats[name + '_padding_fraction'] = (1 - stats[used]/stats[slots]) if stats[slots] else 0.
    stats['shapes'] = {}
    for row in rows:
        for shape, count in row.get('shapes', {}).items():
            stats['shapes'][shape] = stats['shapes'].get(shape, 0) + count
    return stats


def _merge_best(q, owner, valid, offset, best, indices, finite):
    size = best.shape[0]
    segment = jnp.where(valid, owner, size)
    maxima = jax.ops.segment_max(q, segment, size, indices_are_sorted=True)
    position = jnp.arange(q.shape[0], dtype=jnp.int32) + offset
    candidate = jnp.where(valid & (q == maxima[owner]), position, jnp.iinfo(jnp.int32).max)
    first = jax.ops.segment_min(candidate, segment, size, indices_are_sorted=True)
    take = (maxima > best) | ((maxima == best) & (first < indices))
    return (jnp.maximum(best, maxima), jnp.where(take, first, indices),
            finite & jnp.all(~valid | jnp.isfinite(q)))


class Inference:
    def __init__(self, mc, batch=64, action_chunk=2048, *, bucket_batch=False,
                 history_buckets=False, fused=False, attention_impl="manual"):
        if batch < 1 or action_chunk < 1:
            raise ValueError('Inference batch and action chunk must be positive')
        if attention_impl not in ('manual', 'xla', 'cudnn'):
            raise ValueError('attention_impl must be manual, xla or cudnn')
        self.mc, self.batch, self.chunk = mc, batch, action_chunk
        self.bucket_batch = bucket_batch
        self.history_buckets, self.fused = history_buckets, fused
        self.attention_impl = attention_impl
        self.encoder = jax.jit(lambda p, t, l, s: encode_state(
            p, t, l, s, mc, attention_impl=attention_impl)[0])

        def scorer(p, ctx, actions, owner, roles, valid):
            q = q_values(p, ctx[owner], actions, roles, mc)
            return jnp.where(valid, q, -jnp.inf)
        self.scorer = jax.jit(scorer)

        def selector(p, ctx, actions, owner, roles, valid, offset, best, indices, finite):
            return _merge_best(scorer(p, ctx, actions, owner, roles, valid), owner,
                               valid, offset, best, indices, finite)
        self.selector = jax.jit(selector)

        def fused_selector(p, tokens, lengths, state, actions, owner, roles, valid):
            ctx = encode_state(p, tokens, lengths, state, mc,
                               attention_impl=attention_impl)[0]
            return selector(p, ctx, actions, owner, roles, valid, jnp.int32(0),
                            jnp.full(tokens.shape[0], -jnp.inf, jnp.float32),
                            jnp.full(tokens.shape[0], jnp.iinfo(jnp.int32).max, jnp.int32),
                            jnp.bool_(True))
        self.fused_selector = jax.jit(fused_selector)
        self.last_stats = {}

    def _batch_size(self, size, grouped=False):
        bins = (32, 64, 128, 256) if grouped else (8, 16, 32, 64, 128, 256)
        if grouped or self.bucket_batch:
            return next((b for b in bins if size <= b <= self.batch), self.batch)
        return self.batch

    def _plans(self, requests):
        lengths = (requests.data['lengths'] if isinstance(requests, PackedRequests) else
                   np.asarray([len(r['tokens']) for r in requests], np.int32))
        if np.any(lengths < 1) or np.any(lengths > self.mc.max_seq):
            raise ValueError('Invalid inference history length')
        buckets = np.asarray((64, 128, 256, 512), np.int32)
        bucket_ids = np.searchsorted(buckets, lengths)
        for start in range(0, len(requests), self.batch):
            ids = np.arange(start, min(start+self.batch, len(requests)))
            batch = self._batch_size(len(ids))
            length = int(buckets[bucket_ids[ids].max()])
            plans = [(ids, batch, length)]
            if self.history_buckets:
                parts = [ids[bucket_ids[ids] == k] for k in range(4)]
                grouped = [(part, self._batch_size(len(part), grouped=True), int(buckets[k]))
                           for k, part in enumerate(parts) if len(part)]
                # More calls are not worthwhile when even padded token work cannot shrink.
                if sum(b*t for _, b, t in grouped) < batch*length:
                    plans = grouped
            yield from plans

    def _batches(self, requests):
        # Research reordering keeps its existing path; production uses contiguous views.
        if isinstance(requests, PackedRequests) and self.history_buckets:
            requests = list(requests)
        packed = isinstance(requests, PackedRequests)
        stats = []
        for ids, batch, length in self._plans(requests):
            req = requests[int(ids[0]):int(ids[-1])+1] if packed else [requests[i] for i in ids]
            size = len(req)
            tokens = np.zeros((batch, length), np.int32)
            tokens[:, 0] = 1
            lengths = np.ones(batch, np.int32)
            state = np.zeros((batch, STATE_DIM), np.float32)
            if packed:
                lengths[:size] = req.data['lengths']
                width = min(length, req.data['tokens'].shape[1])
                tokens[:size, :width] = req.data['tokens'][:, :width]
                state[:size] = req.data['state']
                counts, actions, role = (req.data[k] for k in ('counts', 'actions', 'role'))
            else:
                for i, r in enumerate(req):
                    lengths[i] = len(r['tokens'])
                    if lengths[i] < 1:
                        raise ValueError('Empty inference history')
                    tokens[i, :lengths[i]] = r['tokens']
                    state[i] = r['state']
                counts = np.asarray([len(r['actions']) for r in req], np.int32)
                actions = np.concatenate([r['actions'] for r in req])
                role = np.asarray([r['role'] for r in req], np.int32)
            if self.attention_impl == 'cudnn':
                validate_attention_tokens(tokens, lengths)
            if np.any(counts < 1):
                raise ValueError('A live decision must have at least one legal action')
            owner = np.repeat(np.arange(size, dtype=np.int32), counts)
            roles = np.repeat(role, counts)
            stats.append({'requests': size, 'legal_actions': len(actions),
                          'action_slots': ((len(actions)+self.chunk-1)//self.chunk)*self.chunk,
                          'max_legal_actions': int(counts.max()),
                          'history_tokens': int(lengths[:size].sum()),
                          'history_slots': batch*length, 'state_slots': batch,
                          'encoder_batches': 1, 'shapes': {f'{batch}x{length}': 1}})
            yield ids, tokens, lengths, state, counts, actions, owner, roles
        self.last_stats = merge_inference_stats(stats)

    def _chunks(self, actions, owner, roles):
        for off in range(0, len(actions), self.chunk):
            m = min(self.chunk, len(actions)-off)
            aa = np.zeros((self.chunk, ACTION_DIM), np.float32)
            oo = np.zeros(self.chunk, np.int32)
            rr = np.zeros(self.chunk, np.int32)
            aa[:m], oo[:m], rr[:m] = actions[off:off+m], owner[off:off+m], roles[off:off+m]
            yield off, m, aa, oo, rr, np.arange(self.chunk) < m

    def score(self, p, requests):
        result = [None] * len(requests)
        for ids, tokens, lengths, state, counts, actions, owner, roles in self._batches(requests):
            ctx = self.encoder(p, tokens, lengths, state)
            scores = np.empty(len(actions), np.float32)
            for off, m, aa, oo, rr, valid in self._chunks(actions, owner, roles):
                scores[off:off+m] = np.asarray(self.scorer(p, ctx, aa, oo, rr, valid))[:m]
            if not np.isfinite(scores).all():
                raise FloatingPointError('Non-finite legal-action scores')
            offsets = np.r_[0, np.cumsum(counts)]
            for i, a, b in zip(ids, offsets[:-1], offsets[1:]):
                result[i] = scores[a:b].copy()
        return result

    def choose(self, p, requests, rng=None, epsilon=0.0):
        if not 0 <= epsilon <= 1 or (epsilon and rng is None):
            raise ValueError('Exploration requires epsilon in [0, 1] and an RNG')
        outputs, offsets, original_ids = [], [], []
        fused_batches = 0
        for ids, tokens, lengths, state, counts, actions, owner, roles in self._batches(requests):
            if self.fused and len(actions) <= self.chunk:
                _, _, aa, oo, rr, valid = next(self._chunks(actions, owner, roles))
                _, indices, finite = self.fused_selector(p, tokens, lengths, state, aa, oo, rr, valid)
                fused_batches += 1
            else:
                # Overflow retains streaming chunks: never truncate or re-encode per chunk.
                ctx = self.encoder(p, tokens, lengths, state)
                best = np.full(ctx.shape[0], -np.inf, np.float32)
                indices = np.full(ctx.shape[0], np.iinfo(np.int32).max, np.int32)
                finite = np.bool_(True)
                for off, _, aa, oo, rr, valid in self._chunks(actions, owner, roles):
                    best, indices, finite = self.selector(p, ctx, aa, oo, rr, valid,
                                                           np.int32(off), best, indices, finite)
            outputs.append((indices, finite))
            offsets.append(np.r_[0, np.cumsum(counts)])
            original_ids.append(ids)
        # All chunks stay on-device. One synchronization returns indices and validity.
        result = [None] * len(requests)
        for (indices, finite), bounds, ids in zip(jax.device_get(outputs), offsets, original_ids):
            if not finite:
                raise FloatingPointError('Non-finite legal-action scores')
            for i, choice in zip(ids, indices[:len(bounds)-1] - bounds[:-1]):
                result[i] = int(choice)
        self.last_stats['fused_batches'] = fused_batches
        if rng is not None:
            # Retain the old interleaved RNG draws, including epsilon == 0.
            counts = (requests.data['counts'] if isinstance(requests, PackedRequests) else
                      [len(r['actions']) for r in requests])
            result = [int(rng.integers(n)) if rng.random() < epsilon else i
                      for i, n in zip(result, counts)]
        return result
