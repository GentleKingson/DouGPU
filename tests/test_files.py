import hashlib
import json
import subprocess
import sys

import pytest

from dougpu import files
from dougpu.checkpoint import atomic_bytes, sha256_file
from dougpu.preflight import _write_report
from tune_local import sha256, write_json


def test_shared_output_hashes_and_dependency_boundary(tmp_path):
    assert atomic_bytes is files.atomic_bytes
    assert sha256_file is sha256 is files.sha256_file
    assert write_json is files.write_json
    path = tmp_path / 'nested' / 'report.json'
    value = {'text': '\u8bad\u7ec3', 'rows': [1, None]}
    _write_report(path, value)
    assert path.read_bytes() == (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {'bad': float('nan')})
    assert path.read_bytes() == before
    _write_report('', value)
    for data in (b'', b'x' * (9 * 1024 * 1024)):
        atomic_bytes(path, data)
        assert sha256(path) == hashlib.sha256(data).hexdigest()
    assert not list(path.parent.glob('*.partial'))
    subprocess.run([sys.executable, '-S', '-c',
                    'import tune_local; import sys; '
                    'assert "numpy" not in sys.modules and "jax" not in sys.modules'], check=True)


@pytest.mark.parametrize('failure', ['fsync', 'replace'])
def test_atomic_failure_keeps_destination_and_cleans_temporary(tmp_path, monkeypatch, failure):
    path = tmp_path / 'report.json'
    path.write_bytes(b'old')
    calls = []
    fsync, replace = files.os.fsync, files.os.replace

    def sync(fd):
        calls.append('fsync')
        if failure == 'fsync':
            raise OSError('injected fsync failure')
        return fsync(fd)

    def commit(src, dst):
        calls.append('replace')
        assert calls == ['fsync', 'replace']
        if failure == 'replace':
            raise OSError('injected replace failure')
        return replace(src, dst)

    monkeypatch.setattr(files.os, 'fsync', sync)
    monkeypatch.setattr(files.os, 'replace', commit)
    with pytest.raises(OSError):
        write_json(path, {'new': True})
    assert path.read_bytes() == b'old'
    assert list(tmp_path.iterdir()) == [path]
