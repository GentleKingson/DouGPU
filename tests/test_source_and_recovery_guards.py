"""Guard actual import bytes and fail closed when checkpoint markers are damaged."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bootstrap import (PINNED_COMMIT, prepare, verify_local_source,
                       verify_upstream_files)
from dougpu.checkpoint import Store


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path/'project'
    vendor = root/'vendor'
    files = {'LICENSE': b'test license\n', 'douzero/__init__.py': b'',
             'douzero/env/__init__.py': b'', 'douzero/env/utils.py': b'VALUE = 7\n'}
    for name, data in files.items():
        path = vendor/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root/'dougpu').mkdir()
    (root/'dougpu'/'__init__.py').write_text('VERSION = 1\n')
    lock = {'engine': 'douzero', 'encoding_schema': 1, 'douzero_commit': PINNED_COMMIT,
            'upstream_files': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
            'doutpu_sha256': 'historical'}
    return root, vendor, lock


def test_locked_source_accepts_normal_bytecode_cache(source_tree):
    root, vendor, lock = source_tree
    cache = vendor/'douzero'/'env'/'__pycache__'
    cache.mkdir()
    (cache/'utils.cpython-312.pyc').write_bytes(b'normal cache is not an importable sibling')
    (vendor/'douzero'/'README.txt').write_text('Non-code documentation is allowed.\n')
    verify_upstream_files(vendor, lock)
    verify_local_source(root, lock)


@pytest.mark.parametrize('relative', [
    'douzero/env/utils/__init__.py',  # This package would shadow locked utils.py.
    'douzero/env/untracked.py',
    'douzero/env/untracked.pyc',
    'douzero/env/untracked.cpython-312-x86_64-linux-gnu.so',
])
def test_untracked_importable_source_is_rejected(source_tree, relative):
    _, vendor, lock = source_tree
    extra = vendor/relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b'not in source manifest')
    with pytest.raises(RuntimeError, match='Untracked or missing importable'):
        verify_upstream_files(vendor, lock)


def test_removing_a_source_from_manifest_does_not_skip_validation(source_tree):
    _, vendor, lock = source_tree
    del lock['upstream_files']['douzero/env/utils.py']
    with pytest.raises(RuntimeError, match='Untracked or missing importable'):
        verify_upstream_files(vendor, lock)


@pytest.mark.parametrize('kind', ['file', 'package', 'vendor'])
def test_raw_symlinks_rejected_even_with_identical_resolved_bytes(source_tree, kind):
    _, vendor, lock = source_tree
    if kind == 'file':
        path = vendor/'douzero'/'env'/'utils.py'
        target = vendor/'original-utils.txt'
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif kind == 'package':
        path = vendor/'douzero'/'env'
        target = vendor/'original-env'
        path.rename(target)
        path.symlink_to(target, target_is_directory=True)
    else:
        target = vendor.with_name('original-vendor')
        vendor.rename(target)
        vendor.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match='symlink'):
        verify_upstream_files(vendor, lock)


@pytest.mark.parametrize('key,value', [
    ('engine', 'reference'), ('engine', None),
    ('encoding_schema', 2), ('encoding_schema', None),
])
def test_local_source_checks_engine_and_encoding_schema(source_tree, key, value):
    root, _, lock = source_tree
    lock[key] = value
    with pytest.raises(RuntimeError, match='engine or encoding schema'):
        verify_local_source(root, lock)


def test_bundled_real_source_manifest_still_verifies():
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root/'upstream_cache'/'source_lock.json').read_text())
    verify_upstream_files(root/'vendor', lock)


INVALID_MARKERS = [
    b'{"file":',
    {},
    {'created_ns': 9, 'sha256': 'a'*64},
    {'file': 'ckpt_missing.zip', 'sha256': 'a'*64},
    {'file': 'ckpt_missing.zip', 'created_ns': 9},
    {'file': '../outside.zip', 'created_ns': 9, 'sha256': 'a'*64},
    {'file': ['ckpt_missing.zip'], 'created_ns': 9, 'sha256': 'a'*64},
    {'file': 'ckpt_missing.zip', 'created_ns': 'newest', 'sha256': 'a'*64},
    {'file': 'ckpt_missing.zip', 'created_ns': True, 'sha256': 'a'*64},
    {'file': 'ckpt_missing.zip', 'created_ns': -1, 'sha256': 'a'*64},
    [],
    None,
]


def put_invalid_marker(directory, document):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'ckpt_invalid.ok.json'
    path.write_bytes(document if isinstance(document, bytes) else json.dumps(document).encode())
    return path


def save_small_checkpoint(store, cycle=1):
    params = {'w': np.array([.1, .2], np.float32)}
    optimizer = {'step': np.array(cycle, np.int32),
                 'm': {'w': np.zeros(2, np.float32)}, 'v': {'w': np.ones(2, np.float32)}}
    return store.save(params, optimizer, params, None,
                      {'cycle': cycle, 'updates': cycle, 'model': {}, 'champion_cycle': 0})


@pytest.mark.parametrize('document', INVALID_MARKERS)
@pytest.mark.parametrize('location', ['local', 'mirror'])
def test_observed_invalid_markers_never_mean_fresh_run(tmp_path, document, location):
    local, mirror = tmp_path/'local', tmp_path/'mirror'
    marker = put_invalid_marker(local if location == 'local' else mirror, document)
    before = marker.read_bytes()
    with pytest.raises(RuntimeError, match='not restarting silently'):
        Store(local, mirror).load_latest()
    assert marker.read_bytes() == before  # Keep the damaged generation for diagnosis.


def test_no_markers_is_still_a_new_run(tmp_path):
    assert Store(tmp_path/'local', tmp_path/'mirror').load_latest() is None


@pytest.mark.parametrize('document', INVALID_MARKERS)
@pytest.mark.parametrize('location', ['local', 'mirror'])
def test_invalid_marker_still_allows_older_valid_generation(tmp_path, document, location):
    store = Store(tmp_path/'local', tmp_path/'mirror', keep=1)
    first = save_small_checkpoint(store)
    marker = put_invalid_marker(store.local if location == 'local' else store.remote, document)
    before = marker.read_bytes()
    restored = store.load_latest()
    assert Path(restored['path']) == first
    assert restored['meta']['updates'] == int(restored['optimizer']['step']) == 1
    second = save_small_checkpoint(store, cycle=2)
    assert Path(store.load_latest()['path']) == second
    assert store.last_remote_ok is True
    assert marker.read_bytes() == before
    assert not first.exists() and not (store.remote/first.name).exists()


@pytest.mark.parametrize('created_ns', ['newest', None, True, -1])
@pytest.mark.parametrize('location', ['local', 'mirror'])
def test_invalid_timestamp_does_not_break_save_or_pruning(tmp_path, created_ns, location):
    store = Store(tmp_path/'local', tmp_path/'mirror', keep=1)
    first = save_small_checkpoint(store)
    document = json.loads(first.with_suffix('.ok.json').read_text())
    document['created_ns'] = created_ns
    marker = put_invalid_marker(store.local if location == 'local' else store.remote, document)
    before = marker.read_bytes()
    second = save_small_checkpoint(store, cycle=2)
    assert Path(store.load_latest()['path']) == second
    assert store.last_remote_ok is True
    assert marker.read_bytes() == before
    assert not first.exists() and not (store.remote/first.name).exists()


def test_corrupt_new_archive_and_marker_fall_back_to_older_mirror(tmp_path):
    local, mirror = tmp_path/'local', tmp_path/'mirror'
    mirrored = save_small_checkpoint(Store(mirror), cycle=1)
    newest = save_small_checkpoint(Store(local), cycle=2)
    newest.write_bytes(b'damaged newer archive')
    put_invalid_marker(local, b'{broken marker')
    restored = Store(local, mirror).load_latest()
    assert Path(restored['path']) == mirrored
    assert restored['meta']['cycle'] == 1


def test_prepare_reuses_old_lock_and_copies_only_rule_identity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cache = root/'upstream_cache'
    cache_before = (cache/'source_lock.json').read_bytes()
    project, run = tmp_path/'project', tmp_path/'run'
    lock = prepare(project, run, upstream_cache=cache)
    assert set(lock) == {'engine', 'encoding_schema', 'douzero_commit', 'archive_sha256', 'upstream_files'}
    assert (cache/'source_lock.json').read_bytes() == cache_before
    assert not (project/'source_lock.json').exists()
    lock['doutpu_sha256'] = 'historical'
    path = run/'source_lock.json'
    path.write_text(json.dumps(lock, indent=4))
    before = path.read_bytes()
    assert prepare(project, run) == lock
    assert path.read_bytes() == before
    (project/'vendor/douzero/env/utils.py').write_text('changed rules')
    with pytest.raises(RuntimeError, match='hash/path mismatch'):
        verify_local_source(project, lock)
    (run/'upstream_source.zip').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='missing/corrupt'):
        prepare(project, run)
