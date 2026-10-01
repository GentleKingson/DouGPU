"""Bounded incremental inference prototype; never used by replay gradients."""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from dougpu.inference import Inference
from dougpu.model import dtype, matmul, dense, rms


def positioned_rope(x, positions):
    dh = x.shape[-1]
    freq = 10000.0 ** (-jnp.arange(dh//2, dtype=jnp.float32) / (dh//2))
    # Full-prefix RoPE constant-folds its angles. Reuse the same absolute
    # position constants rather than a numerically different dynamic sin/cos.
    angle = jnp.arange(512, dtype=jnp.float32)[:, None] * freq[None, :]
    index = jnp.clip(positions, 0, 511)
    co, si = jnp.cos(angle)[index][:, :, None], jnp.sin(angle)[index][:, :, None]
    a, b = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    return jnp.concatenate([a*co-b*si, a*si+b*co], -1).astype(x.dtype)


def append_context(p, cache, suffix, offsets, lengths, slots, state, *, cfg, history_len):
    keys, values, public = cache
    capacity, _, heads, dh = keys[0].shape
    batch, count = suffix.shape
    positions = offsets[:, None] + jnp.arange(count)[None, :]
    valid = (positions < lengths[:, None]) & (slots[:, None] < capacity)
    write_slots = jnp.where(valid, slots[:, None], capacity)
    read_slots = jnp.minimum(slots, capacity-1)
    causal = (jnp.arange(history_len)[None, None, :] <= positions[:, :, None])
    mask = causal & (jnp.arange(history_len)[None, None, :] < lengths[:, None, None])
    x = p['embedding'][suffix].astype(dtype(cfg))
    new_keys, new_values = [], []
    for layer in range(cfg.layers):
        name = f'block{layer}.'
        y = rms(x, p[name+'anorm'])
        shape = (batch, count, heads, dh)
        q = positioned_rope(rms(matmul(y, p[name+'q'], cfg).reshape(shape), p[name+'qnorm']), positions)
        k = positioned_rope(rms(matmul(y, p[name+'k'], cfg).reshape(shape), p[name+'knorm']), positions)
        v = matmul(y, p[name+'v'], cfg).reshape(shape)
        kb = keys[layer].at[write_slots, positions].set(k, mode='drop')
        vb = values[layer].at[write_slots, positions].set(v, mode='drop')
        past_k, past_v = kb[read_slots, :history_len], vb[read_slots, :history_len]
        logits = jnp.einsum('bthd,bshd->bhts', q, past_k,
                            preferred_element_type=jnp.float32,
                            precision=None if cfg.bf16 else jax.lax.Precision.HIGHEST) / np.sqrt(dh)
        probs = jax.nn.softmax(jnp.where(mask[:, None], logits, -1e9), -1).astype(dtype(cfg))
        z = jnp.einsum('bhts,bshd->bthd', probs, past_v,
                       precision=None if cfg.bf16 else jax.lax.Precision.HIGHEST).reshape((batch, count, cfg.width))
        x = x + matmul(z, p[name+'o'], cfg)
        y = rms(x, p[name+'fnorm'])
        y = jax.nn.silu(matmul(y, p[name+'gate'], cfg)) * matmul(y, p[name+'up'], cfg)
        x = x + matmul(y, p[name+'down'], cfg)
        new_keys.append(kb)
        new_values.append(vb)
    x = rms(x, p['final_norm'])
    last = jnp.clip(lengths-offsets-1, 0, count-1)
    context = jnp.where((lengths > offsets)[:, None], x[jnp.arange(batch), last], public[read_slots])
    public = public.at[slots].set(context, mode='drop')
    y = jnp.concatenate([context, state.astype(dtype(cfg))], -1)
    y = jax.nn.silu(dense(p, 'state1', y, cfg))
    y = jax.nn.silu(dense(p, 'state2', y, cfg))
    return (tuple(new_keys), tuple(new_values), public), y


class KVInference(Inference):
    """One immutable policy version, fixed game slots, complete-prefix checks.

    Separate instances are needed for current policy and champion. Only private
    cache buffers are donated, never model/champion parameters. On version or
    parameter-object change every slot rebuilds before its next use.
    """
    def __init__(self, mc, capacity, batch=64, action_chunk=2048, *, bucket_batch=True):
        super().__init__(mc, batch, action_chunk, bucket_batch=bucket_batch)
        if capacity < 1:
            raise ValueError('Cache capacity must be positive')
        self.capacity = capacity
        self.kernel = jax.jit(partial(append_context, cfg=mc),
                              static_argnames=('history_len',), donate_argnums=(1,))
        self.encoder = self._cached_encoder
        self.cache = None
        self.histories = [b''] * capacity
        self.params_ref = self.version = None
        self.cache_stats = {}

    def _allocate(self):
        shape = (self.capacity, self.mc.max_seq, self.mc.heads, self.mc.width//self.mc.heads)
        keys = tuple(jnp.zeros(shape, dtype(self.mc)) for _ in range(self.mc.layers))
        values = tuple(jnp.zeros(shape, dtype(self.mc)) for _ in range(self.mc.layers))
        self.cache = keys, values, jnp.zeros((self.capacity, self.mc.width), dtype(self.mc))

    def _prepare(self, p, requests, cache_keys, policy_version):
        keys = np.asarray(cache_keys)
        if (keys.ndim != 1 or len(keys) != len(requests) or
                (len(keys) and (keys.dtype.kind not in 'iu' or keys.min() < 0 or keys.max() >= self.capacity)) or
                len(np.unique(keys)) != len(keys)):
            raise ValueError('Cache keys must be unique valid game slots')
        if p is not self.params_ref or policy_version != self.version:
            self.histories = [b''] * self.capacity
            self.params_ref, self.version = p, policy_version
        if self.cache is None:
            self._allocate()
        self.call_keys, self.offset = keys, 0
        self.cache_stats = {'appended_tokens': 0, 'reused_tokens': 0, 'reset_slots': 0}

    def _cached_encoder(self, p, tokens, lengths, state):
        batch, history_len = tokens.shape
        keys = self.call_keys[self.offset:self.offset+batch]
        self.offset += len(keys)
        offsets = np.zeros(batch, np.int32)
        slots = np.full(batch, self.capacity, np.int32)
        new_lengths = np.zeros(batch, np.int32)
        snapshots = []
        for i, slot in enumerate(keys):
            raw = tokens[i, :lengths[i]]
            if np.any(raw <= 0) or np.any(raw >= self.mc.vocab):
                raise ValueError('Only non-PAD vocabulary tokens may appear in a cached history')
            current = raw.astype(np.uint8).tobytes()
            before = self.histories[slot]
            if not current.startswith(before):
                before = b''
                self.cache_stats['reset_slots'] += 1
            offsets[i], slots[i], new_lengths[i] = len(before), slot, len(current)
            snapshots.append(current)
        delta = new_lengths - offsets
        count = next(n for n in (8, 16, 32, 64, 128, 256, 512) if n >= max(1, int(delta.max())))
        suffix = np.zeros((batch, count), np.int32)
        for i in range(len(keys)):
            suffix[i, :delta[i]] = tokens[i, offsets[i]:new_lengths[i]]
        self.cache, context = self.kernel(p, self.cache, suffix, offsets, new_lengths,
                                         slots, state, history_len=history_len)
        for slot, current in zip(keys, snapshots):
            self.histories[slot] = current
        self.cache_stats['appended_tokens'] += int(delta.sum())
        self.cache_stats['reused_tokens'] += int(offsets.sum())
        return context

    def _run(self, method, p, requests, cache_keys, policy_version, *args):
        self._prepare(p, requests, cache_keys, policy_version)
        try:
            return method(p, requests, *args)
        except BaseException:
            # A failed donated-buffer dispatch must never leave a reusable cache.
            self.cache = None
            self.histories = [b''] * self.capacity
            raise

    def choose(self, p, requests, rng=None, epsilon=0., *, cache_keys, policy_version):
        return self._run(super().choose, p, requests, cache_keys, policy_version, rng, epsilon)

    def score(self, p, requests, *, cache_keys, policy_version):
        return self._run(super().score, p, requests, cache_keys, policy_version)
