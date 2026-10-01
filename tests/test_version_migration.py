import json
from dataclasses import asdict
import numpy as np
import pytest
from dougpu.checkpoint import Store, sha256_file
from dougpu.config import ModelConfig, TrainConfig
from dougpu.replay import Replay
from fork_run import fork


@pytest.mark.parametrize('case', ['valid', 'missing', 'skipped', 'reverse', 'overflow'])
def test_version_migration_proves_history_and_preserves_source(tmp_path, case):
    mc = ModelConfig()
    tc = TrainConfig(replay_capacity=3, replay_version_updates=(case == 'reverse'))
    replay = Replay(3)
    replay.add([(np.ones(6, np.uint8), np.zeros(21), np.zeros(16), role,
                 np.zeros(30), 1., role+1) for role in range(3)])
    if case == 'overflow':
        replay.data['version'][0] = np.iinfo(np.int32).max//4+1
    rows = [{'event': 'train', 'cycle': i, 'updates': 4*i, 'successful_steps': 4} for i in (1, 2, 3)]
    if case == 'missing':
        rows.pop(1)
    if case == 'skipped':
        rows[1]['successful_steps'] = 3
    log = tmp_path/'origin.jsonl'
    log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    lock = {'engine': 'douzero', 'encoding_schema': 1, 'douzero_commit': 'pinned',
            'upstream_files': {'rule.py': 'hash'}, 'doutpu_sha256': 'old'}
    new_lock = dict(lock, doutpu_sha256='new')
    params = {'w': np.array([[.1, .2]], np.float32)}
    opt = {'step': np.array(12, np.int32), 'm': {'w': np.ones((1, 2), np.float32)},
           'v': {'w': np.ones((1, 2), np.float32)*2}}
    meta = {'model': asdict(mc), 'train': asdict(tc), 'source_lock': lock,
            'cycle': 3, 'updates': 12, 'champion_cycle': 0}
    source = tmp_path/'source'
    archive = Store(source).save(params, opt, params, replay, meta, log)
    digest = sha256_file(archive)
    config = {'model': asdict(mc), 'train': dict(asdict(tc), sample_credit=True,
                                               replay_version_updates=(case != 'reverse'))}
    target = tmp_path/'target'/'state'
    if case != 'valid':
        with pytest.raises(ValueError):
            fork(source, tmp_path/'work', target, config, new_lock)
        assert Store(target).load_latest() is None
    else:
        fork(source, tmp_path/'work', target, config, new_lock)
        saved = Store(target).load_latest()
        assert saved['meta']['updates'] == int(saved['optimizer']['step']) == 12
        np.testing.assert_array_equal(saved['replay']['version'], [4, 8, 12])
        np.testing.assert_array_equal(saved['optimizer']['m']['w'], opt['m']['w'])
        assert saved['meta']['forked_from']['source_sha256'] == digest
    assert sha256_file(archive) == digest
    np.testing.assert_array_equal(Store(source).load_latest()['replay']['version'], replay.data['version'])
