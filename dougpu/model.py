"""Small Llama-like causal encoder implemented directly in stable JAX primitives.

Parameters and optimizer moments are FP32. Matmuls/activations may use BF16;
attention softmax, normalizations, Q outputs and losses remain FP32.
"""
from functools import partial
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from .encoding import STATE_DIM, ACTION_DIM, BELIEF_DIM


def init_params(cfg, seed=0):
    cfg.validate()
    rng = np.random.default_rng(seed)
    p = {}
    d = cfg.width
    def matrix(name, ni, no, scale=1.0):
        p[name] = (rng.standard_normal((ni, no)) * (scale / np.sqrt(ni))).astype(np.float32)
    def dense(name, ni, no, scale=1.0):
        matrix(name + ".w", ni, no, scale)
        p[name + ".b"] = np.zeros(no, np.float32)
    p["embedding"] = (0.03 * rng.standard_normal((cfg.vocab, d))).astype(np.float32)
    for i in range(cfg.layers):
        b = f"block{i}."
        for name in ("anorm", "fnorm"):
            p[b+name] = np.ones(d, np.float32)
        for name in ("qnorm", "knorm"):
            p[b+name] = np.ones(d // cfg.heads, np.float32)
        for name in ("q", "k", "v", "o"):
            matrix(b+name, d, d)
        matrix(b+"gate", d, cfg.ffn)
        matrix(b+"up", d, cfg.ffn)
        matrix(b+"down", cfg.ffn, d, 1 / np.sqrt(2 * cfg.layers))
    p["final_norm"] = np.ones(d, np.float32)
    dense("state1", d + STATE_DIM, d)
    dense("state2", d, d)
    dense("q1", d + ACTION_DIM, cfg.q_hidden)
    dense("q2", cfg.q_hidden, d)
    dense("qout", d, 3, 0.01)
    matrix("ntp", d, cfg.vocab)
    dense("belief1", d, d)
    dense("belief2", d, BELIEF_DIM)
    return jax.device_put(p)


def dtype(cfg):
    return jnp.bfloat16 if cfg.bf16 else jnp.float32


def matmul(x, w, cfg):
    return jnp.matmul(x.astype(dtype(cfg)), w.astype(dtype(cfg)),
                      precision=None if cfg.bf16 else jax.lax.Precision.HIGHEST)


def dense(p, name, x, cfg):
    return (matmul(x, p[name + ".w"], cfg) + p[name + ".b"].astype(dtype(cfg)))


def rms(x, weight):
    y = x.astype(jnp.float32)
    y = y * jax.lax.rsqrt(jnp.mean(y*y, axis=-1, keepdims=True) + 1e-6)
    return (y * weight).astype(x.dtype)


def rope(x):
    # [B, T, H, Dh], positions are independent of padding length.
    t, dh = x.shape[1], x.shape[-1]
    freq = 10000.0 ** (-jnp.arange(dh//2, dtype=jnp.float32) / (dh//2))
    angle = jnp.arange(t, dtype=jnp.float32)[:, None] * freq[None, :]
    co, si = jnp.cos(angle)[None, :, None, :], jnp.sin(angle)[None, :, None, :]
    a, b = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    return jnp.concatenate([a*co-b*si, a*si+b*co], axis=-1).astype(x.dtype)


def block(p, x, tokens, *, cfg, layer, attention_impl="manual"):
    b = f"block{layer}."
    y = rms(x, p[b+"anorm"])
    bs, t, d = y.shape
    shape = (bs, t, cfg.heads, d//cfg.heads)
    q = rope(rms(matmul(y, p[b+"q"], cfg).reshape(shape), p[b+"qnorm"]))
    k = rope(rms(matmul(y, p[b+"k"], cfg).reshape(shape), p[b+"knorm"]))
    v = matmul(y, p[b+"v"], cfg).reshape(shape)
    if attention_impl == "manual":
        # Keep the notebook's reference arithmetic unchanged for checkpoint
        # migration and numerical comparisons.
        logits = jnp.einsum("bthd,bshd->bhts", q, k, preferred_element_type=jnp.float32,
                            precision=None if cfg.bf16 else jax.lax.Precision.HIGHEST) / np.sqrt(d//cfg.heads)
        causal = jnp.arange(t)[:, None] >= jnp.arange(t)[None, :]
        mask = causal[None, None, :, :] & (tokens[:, None, None, :] != 0)
        probs = jax.nn.softmax(jnp.where(mask, logits, -1e9), axis=-1).astype(dtype(cfg))
        z = jnp.einsum("bhts,bshd->bthd", probs, v,
                       precision=None if cfg.bf16 else jax.lax.Precision.HIGHEST).reshape((bs, t, d))
    elif attention_impl == "xla":
        # XLA retains the original PAD-key semantics, even for interior PADs.
        with jax.default_matmul_precision('default' if cfg.bf16 else 'float32'):
            z = jax.nn.dot_product_attention(
                q, k, v, mask=tokens[:, None, None, :] != 0, is_causal=True,
                implementation="xla").reshape((bs, t, d))
    elif attention_impl == "cudnn":
        if not cfg.bf16:
            raise ValueError("cuDNN attention requires ModelConfig.bf16=True")
        # CPU entry points validate a nonempty, contiguous non-PAD prefix.
        # The length API avoids materializing a [B, T, T] additive bias and
        # skips unused PAD queries. Valid token outputs and losses retain the
        # notebook semantics; raw hidden states at PAD positions may differ.
        lengths = jnp.sum(tokens != 0, axis=-1, dtype=jnp.int32)
        # Select the kernel explicitly. Unsupported devices/shapes/library
        # versions raise, with no silent manual/XLA fallback. Softmax uses
        # FP32 accumulation, while Q/K/V and attention outputs remain BF16.
        z = jax.nn.dot_product_attention(
            q, k, v, is_causal=True, query_seq_lengths=lengths,
            key_value_seq_lengths=lengths,
            implementation="cudnn").reshape((bs, t, d))
    else:
        raise ValueError("attention_impl must be manual, xla or cudnn")
    x = x + matmul(z, p[b+"o"], cfg)
    y = rms(x, p[b+"fnorm"])
    y = jax.nn.silu(matmul(y, p[b+"gate"], cfg)) * matmul(y, p[b+"up"], cfg)
    return x + matmul(y, p[b+"down"], cfg)


def encode_history(p, tokens, lengths, cfg, *, attention_impl="manual"):
    x = p["embedding"][tokens].astype(dtype(cfg))
    for i in range(cfg.layers):
        fn = partial(block, cfg=cfg, layer=i, attention_impl=attention_impl)
        x = (jax.checkpoint(fn) if cfg.remat else fn)(p, x, tokens)
    x = rms(x, p["final_norm"])
    ctx = x[jnp.arange(x.shape[0]), jnp.maximum(lengths - 1, 0)]
    return ctx, x


def encode_state(p, tokens, lengths, state, cfg, *, attention_impl="manual"):
    public_ctx, hidden = encode_history(p, tokens, lengths, cfg,
                                        attention_impl=attention_impl)
    y = jnp.concatenate([public_ctx, state.astype(dtype(cfg))], -1)
    y = jax.nn.silu(dense(p, "state1", y, cfg))
    y = jax.nn.silu(dense(p, "state2", y, cfg))
    return y, hidden


def q_values(p, ctx, actions, roles, cfg):
    x = jnp.concatenate([ctx, actions.astype(dtype(cfg))], -1)
    x = jax.nn.silu(dense(p, "q1", x, cfg))
    x = jax.nn.silu(dense(p, "q2", x, cfg))
    # Keep the final scalar regression projection in FP32 as well as its loss.
    all_roles = (jnp.matmul(x.astype(jnp.float32), p["qout.w"],
                            precision=jax.lax.Precision.HIGHEST) + p["qout.b"])
    return jnp.take_along_axis(all_roles, roles[..., None], axis=-1)[..., 0]


def sample_losses(p, batch, cfg, tc):
    ctx, h = encode_state(p, batch["tokens"], batch["lengths"], batch["state"], cfg,
                          attention_impl=getattr(tc, "attention_impl", "manual"))
    q = q_values(p, ctx, batch["actions"], batch["role"], cfg)
    qerr = jnp.square(q - batch["target"])
    w = batch["weight"]
    qloss = jnp.mean(qerr * w)
    ntp = jnp.array(0., jnp.float32)
    if tc.ntp_weight > 0:
        # Shifted targets; no future token is visible through causal attention.
        logit = matmul(h[:, :-1], p["ntp"], cfg).astype(jnp.float32)
        target = batch["tokens"][:, 1:]
        mask = target != 0
        ce = -jnp.take_along_axis(jax.nn.log_softmax(logit), target[..., None], axis=-1)[..., 0]
        per_sequence = jnp.sum(ce * mask, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1), 1)
        ntp = jnp.mean(per_sequence * w)
    belief = jnp.array(0., jnp.float32)
    if tc.belief_weight > 0:
        y = jax.nn.silu(dense(p, "belief1", ctx, cfg))
        pred = jax.nn.sigmoid(dense(p, "belief2", y, cfg).astype(jnp.float32))
        belief = jnp.mean(jnp.mean(jnp.square(pred - batch["belief"]), -1) * w)
    total = qloss + tc.ntp_weight * ntp + tc.belief_weight * belief
    return total, jnp.stack([total, qloss, ntp, belief])


def init_optimizer(p):
    return {"step": jnp.array(0, jnp.int32),
            "m": jax.tree_util.tree_map(jnp.zeros_like, p),
            "v": jax.tree_util.tree_map(jnp.zeros_like, p)}


def make_train_step(cfg, tc):
    if tc.learner_remat is not None:
        cfg = replace(cfg, remat=tc.learner_remat)
    value_grad = jax.value_and_grad(lambda p, b: sample_losses(p, b, cfg, tc), has_aux=True)
    def step(p, opt, batches):
        zero = jax.tree_util.tree_map(jnp.zeros_like, p)
        def scan(carry, b):
            gs, ls = carry
            (_, metrics), grad = value_grad(p, b)
            return (jax.tree_util.tree_map(jnp.add, gs, grad), ls+metrics), None
        groups = (batches,) if tc.history_groups == 1 else batches
        carry = zero, jnp.zeros(4, jnp.float32)
        for group in groups:
            carry, _ = jax.lax.scan(scan, carry, group)
        grads, metrics = carry
        k = sum(group['target'].shape[0] for group in groups)
        grads = jax.tree_util.tree_map(lambda g: g/k, grads)
        metrics = metrics/k
        norm = jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree_util.tree_leaves(grads)))
        valid = jnp.isfinite(norm) & jnp.all(jnp.isfinite(metrics))
        grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1., tc.grad_clip/(norm+1e-6)), grads)
        t = opt["step"] + 1
        m = jax.tree_util.tree_map(lambda old, g: .9*old + .1*g, opt["m"], grads)
        v = jax.tree_util.tree_map(lambda old, g: .999*old + .001*g*g, opt["v"], grads)
        new = {name: w - tc.lr * ((m[name]/(1-.9**t)) /
                    (jnp.sqrt(v[name]/(1-.999**t))+1e-8) +
                    (tc.weight_decay*w if w.ndim >= 2 else 0.)) for name, w in p.items()}
        nopt = {"step": t, "m": m, "v": v}
        p, opt = jax.lax.cond(valid, lambda: (new, nopt), lambda: (p, opt))
        return p, opt, jnp.concatenate([metrics, jnp.array([norm, valid.astype(jnp.float32)])])
    # No donated buffers: champion snapshots may intentionally share immutable arrays.
    return jax.jit(step)
