import numpy as np
import jax
import pytest
from dataclasses import replace
from dougpu.config import ModelConfig, TrainConfig
from dougpu.model import init_params, init_optimizer, make_train_step
from dougpu.replay import group_history


@pytest.mark.parametrize('groups', [2, 4])
def test_same_rows_role_weights_sequence_losses_and_single_adam(groups):
    rng = np.random.default_rng(38)
    cfg = ModelConfig(width=16, layers=1, heads=2, ffn=32, q_hidden=16, bf16=False)
    tc = TrainConfig(micro_batch=4, accumulation=4)
    lengths = np.array([6, 63, 128, 129, 64, 257, 9, 48, 256, 17, 70, 8, 128, 3, 81, 4])
    tokens = rng.integers(1, 23, (16, 512), dtype=np.int32)
    tokens[np.arange(512)[None, :] >= lengths[:, None]] = 0
    roles = np.arange(16) % 3
    flat = {'tokens': tokens, 'lengths': lengths.astype(np.int32),
            'role': roles.astype(np.int32), 'state': rng.random((16, 21), dtype=np.float32),
            'actions': rng.random((16, 16), dtype=np.float32),
            'belief': rng.random((16, 30), dtype=np.float32), 'target': np.ones(16, np.float32),
            'weight': (16/(3*np.bincount(roles)[roles])).astype(np.float32)}
    batch = {key: value.reshape((4, 4)+value.shape[1:]) for key, value in flat.items()}
    grouped = group_history(batch, groups)
    order = np.argsort(lengths, kind='stable')
    for key in flat:
        if key == 'tokens':
            restored = np.concatenate([np.pad(g[key].reshape(-1, g[key].shape[-1]),
                                              ((0, 0), (0, 512-g[key].shape[-1]))) for g in grouped])
        else:
            restored = np.concatenate([g[key].reshape((-1,)+flat[key].shape[1:]) for g in grouped])
        np.testing.assert_array_equal(restored, flat[key][order])
    p = init_params(cfg, 11)
    opt = init_optimizer(p)
    pa, oa, ma = make_train_step(cfg, tc)(p, opt, jax.device_put(batch))
    step = make_train_step(cfg, replace(tc, history_groups=groups))
    pb, ob, mb = step(p, opt, jax.device_put(grouped))
    assert int(oa['step']) == int(ob['step']) == 1
    np.testing.assert_allclose(ma, mb, rtol=3e-5, atol=2e-6)
    for a, b in zip(jax.tree_util.tree_leaves((pa, oa)), jax.tree_util.tree_leaves((pb, ob))):
        np.testing.assert_allclose(a, b, rtol=5e-5, atol=5e-7)
    broken = list(grouped)
    broken[-1] = dict(broken[-1], target=np.full_like(broken[-1]['target'], np.nan))
    pc, oc, metrics = step(p, opt, jax.device_put(tuple(broken)))
    assert np.asarray(metrics)[-1] == 0 and int(oc['step']) == 0
    for a, b in zip(jax.tree_util.tree_leaves((p, opt)), jax.tree_util.tree_leaves((pc, oc))):
        np.testing.assert_array_equal(a, b)
