import hashlib
import json
import zipfile

import numpy as np
import pytest

from dougpu.checkpoint import Store


@pytest.mark.parametrize('log_data', [None, b'', b'{"event":"train","updates":4}\n' * 500])
def test_log_compression_roundtrip_and_corrupt_fallback(tmp_path, log_data):
    store = Store(tmp_path/'local', tmp_path/'remote')
    params = {'w': np.arange(12, dtype=np.float32).reshape(3, 4)}
    opt = {'step': np.int32(1), 'm': params, 'v': params}
    meta = {'cycle': 1, 'updates': 1, 'model': {}, 'champion_cycle': 0}
    log = tmp_path/'metrics.jsonl'
    if log_data is not None:
        log.write_bytes(log_data)
    first = store.save(params, opt, params, None, meta, log)
    assert store.last_remote_ok
    with zipfile.ZipFile(first) as z:
        manifest = json.loads(z.read('manifest.json'))
        for entry in z.infolist():
            assert entry.compress_type == (zipfile.ZIP_DEFLATED if entry.filename == 'metrics.jsonl'
                                            else zipfile.ZIP_STORED)
        assert all(hashlib.sha256(z.read(k)).hexdigest() == v for k, v in manifest.items())
        assert ('metrics.jsonl' in z.namelist()) == (log_data is not None)
    loaded = Store(tmp_path/'empty', tmp_path/'remote').load_latest()
    assert loaded['log'] == (log_data or b'')
    np.testing.assert_array_equal(loaded['params']['w'], params['w'])
    second = store.save(params, opt, params, None, dict(meta, cycle=2), log)
    for path in (second, tmp_path/'remote'/second.name):
        path.write_bytes(b'broken')
    loaded = store.load_latest()
    assert loaded['meta']['cycle'] == 1 and loaded['log'] == (log_data or b'')

    assert loaded['path'] == str(first)
    assert loaded['sha256'] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert loaded['sha256'] != hashlib.sha256(second.read_bytes()).hexdigest()
