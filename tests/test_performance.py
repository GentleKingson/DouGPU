from dataclasses import asdict
import numpy as np
from dougpu.config import ModelConfig, TrainConfig
from dougpu.encoding import action_features, counts, _action_counts
from dougpu.replay import Replay


def test_action_cache_validation_and_isolation():
    actions = [(), (3,), (3, 3), (3, 3, 3, 3), (20, 30)]
    expected = np.array([np.r_[counts(a), float(not a)] for a in actions], np.float32)
    out = action_features(actions)
    np.testing.assert_array_equal(out, expected)
    out[:] = -9
    np.testing.assert_array_equal(action_features(actions), expected)
    assert not _action_counts((3,)).flags.writeable
    for illegal in ((3,)*5, (20,20), (99,)):
        try:
            action_features([illegal])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid action accepted')


def test_replay_bucket_boundaries():
    cfg = TrainConfig(micro_batch=3, accumulation=1)
    for length, bucket in ((6,64),(64,64),(65,128),(128,128),(129,256),(256,256),(257,512),(486,512)):
        replay = Replay(3)
        replay.add([(np.ones(length,np.uint8), np.zeros(21), np.zeros(16), role,
                     np.zeros(30), 1., 0) for role in range(3)])
        b = replay.sample(cfg,np.random.default_rng(3),0)
        assert b['tokens'].shape == (1,3,bucket)
        assert np.all(b['lengths']==length)
        assert np.all(b['tokens'][...,:length]==1)
        assert not np.any(b['tokens'][...,length:])


def test_padding_preserves_fp32_loss_and_gradients():
    import jax
    from dougpu.model import init_params, sample_losses
    mc = ModelConfig(width=32,layers=1,heads=2,ffn=64,q_hidden=32,bf16=False)
    tc = TrainConfig(micro_batch=3,accumulation=1)
    p = init_params(mc,17)
    rng = np.random.default_rng(91)
    tokens = np.zeros((3,512),np.int32)
    lengths = np.array([6,64,113],np.int32)
    for i,n in enumerate(lengths):
        tokens[i,:n] = rng.integers(1,23,n)
    b = {'tokens':tokens,'lengths':lengths,'state':rng.random((3,21)).astype(np.float32),
         'actions':rng.random((3,16)).astype(np.float32),'role':np.arange(3,dtype=np.int32),
         'belief':rng.random((3,30)).astype(np.float32),'target':np.ones(3,np.float32),
         'weight':np.ones(3,np.float32)}
    fn = jax.jit(jax.value_and_grad(lambda p,b:sample_losses(p,b,mc,tc)[0]))
    a,ga = fn(p,b)
    c,gc = fn(p,dict(b,tokens=tokens[:,:128]))
    np.testing.assert_allclose(a,c,rtol=1e-5,atol=1e-6)
    for key in ga:
        np.testing.assert_allclose(ga[key],gc[key],rtol=2e-4,atol=1e-6)


def test_fork_preserves_source_and_learner(tmp_path):
    from dougpu.checkpoint import Store,sha256_file
    from fork_run import fork
    mc,tc=ModelConfig(),TrainConfig()
    old={'engine':'douzero','encoding_schema':1,'douzero_commit':'fixed',
         'upstream_files':{'rule.py':'hash'},'doutpu_sha256':'original'}
    new=dict(old,doutpu_sha256='optimized')
    p={'w':np.array([[.2,.3]],np.float32)}
    opt={'step':np.array(7,np.int32),'m':{'w':np.ones((1,2),np.float32)},
         'v':{'w':np.ones((1,2),np.float32)*2}}
    meta={'model':asdict(mc),'train':asdict(tc),'source_lock':old,'cycle':9,'updates':7,
          'champion_cycle':1}
    source=tmp_path/'original'/'state'
    snapshot=Store(source).save(p,opt,p,None,meta)
    digest=sha256_file(snapshot)
    cfg={'model':asdict(mc),'train':dict(asdict(tc),workers=16,envs_per_worker=16)}
    target=tmp_path/'new'/'state'
    fork(source,tmp_path/'work',target,cfg,new)
    assert sha256_file(snapshot)==digest
    loaded=Store(target).load_latest()
    np.testing.assert_array_equal(loaded['params']['w'],p['w'])
    np.testing.assert_array_equal(loaded['optimizer']['m']['w'],opt['m']['w'])
    assert int(loaded['optimizer']['step'])==7 and loaded['meta']['cycle']==9
    assert loaded['meta']['source_lock']==new
    try:
        fork(source,tmp_path/'work',target,cfg,new)
    except ValueError:
        pass
    else:
        raise AssertionError('Existing target must not be overwritten')
    try:
        fork(source,tmp_path/'elsewhere',source,cfg,new)
    except ValueError:
        pass
    else:
        raise AssertionError('Source must not be overwritten')
