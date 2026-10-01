"""Atomic file output and hashes without training dependencies."""
from pathlib import Path
import hashlib
import json
import os
import uuid


def sha256_file(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with open(tmp, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, value):
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    atomic_bytes(path, data.encode('utf-8'))
