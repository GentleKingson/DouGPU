"""Pin and cache the upstream rules engine without installing its Torch trainer.

Network is needed on a new experiment. Subsequent runs verify and reuse the exact
archived source. The checked-out git commit, file hashes and package hash are
persisted before training. No branch-tip upgrade is performed during resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
from dougpu.checkpoint import atomic_bytes, copy_verified, json_bytes, sha256_file

REPO = 'https://github.com/kwai/DouZero.git'
PINNED_COMMIT = '718a5c920bf3361e34178a38f3b80458e176b351'


def project_hash(root):
    h = hashlib.sha256()
    root = Path(root)
    paths = list((root/'dougpu').glob('*.py'))
    paths += [root / name for name in ('bootstrap.py', 'fork_run.py', 'run_local.py', 'tune_local.py')
              if (root / name).is_file()]
    for p in sorted(paths):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def safe_extract(path, dest):
    dest = Path(dest).resolve()
    with zipfile.ZipFile(path) as z:
        for member in z.infolist():
            name = member.filename
            out = (dest/name).resolve()
            if out != dest and dest not in out.parents:
                raise ValueError('Unsafe archive path')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Source archive may not contain symlinks')
        z.extractall(dest)


def prepare(root, savedir, commit=PINNED_COMMIT, upstream_cache=None):
    root, savedir = Path(root).resolve(), Path(savedir).resolve()
    savedir.mkdir(parents=True, exist_ok=True)
    vendor = root/'vendor'
    lock_path = savedir/'source_lock.json'
    archive_path = savedir/'upstream_source.zip'
    local_hash = project_hash(root)
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock['doutpu_sha256'] != local_hash:
            raise RuntimeError('Local trainer source changed; use a NEW experiment folder')
        if commit and commit != lock['douzero_commit']:
            raise RuntimeError('Requested engine commit differs from saved experiment')
        if not archive_path.exists() or sha256_file(archive_path) != lock['archive_sha256']:
            raise RuntimeError('Cached upstream archive is missing/corrupt; do not silently change engine versions')
        safe_extract(archive_path, vendor)
    elif upstream_cache:
        # Reuse verified original rule bytes, but give the migrated trainer its own lock.
        cache = Path(upstream_cache).resolve()
        old = json.loads((cache/'source_lock.json').read_text())
        source = cache/'upstream_source.zip'
        if old.get('engine') != 'douzero' or old.get('encoding_schema') != 1:
            raise ValueError('Cached source engine/encoding schema mismatch')
        if old.get('douzero_commit') != commit or sha256_file(source) != old['archive_sha256']:
            raise ValueError('Cached upstream commit or archive hash mismatch')
        safe_extract(source, vendor)
        verify_upstream_files(vendor, old)
        lock = dict(old, doutpu_sha256=local_hash)
        copy_verified(source, archive_path, old['archive_sha256'])
        atomic_bytes(lock_path, json_bytes(lock))
    else:
        # Network failures stop before any training starts. No unpinned fallback.
        with tempfile.TemporaryDirectory(prefix='dougpu_upstream_') as tmp:
            checkout = Path(tmp)/'checkout'
            if commit:
                subprocess.run(['git', 'init', str(checkout)], check=True)
                subprocess.run(['git', '-C', str(checkout), 'remote', 'add', 'origin', REPO], check=True)
                subprocess.run(['git', '-C', str(checkout), 'fetch', '--depth', '1', 'origin', commit], check=True)
                subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
            else:
                subprocess.run(['git', 'clone', '--depth', '1', REPO, str(checkout)], check=True)
            sha = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
            chosen = [checkout/'douzero'/'__init__.py'] + list((checkout/'douzero'/'env').glob('*.py'))
            # Optional baseline evaluator can import the official model on CPU.
            for relative in ('douzero/dmc/__init__.py', 'douzero/dmc/models.py', 'LICENSE', 'NOTICE'):
                p = checkout/relative
                if p.exists():
                    chosen.append(p)
            required = {'game.py', 'move_generator.py', 'move_detector.py', 'move_selector.py', 'utils.py'}
            if not required.issubset({p.name for p in chosen}):
                raise RuntimeError('Upstream layout changed; manual review is required')
            files = {str(p.relative_to(checkout)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in chosen}
            packed = Path(tmp)/'source.zip'
            with zipfile.ZipFile(packed, 'w', zipfile.ZIP_DEFLATED) as z:
                for p in chosen:
                    z.write(p, str(p.relative_to(checkout)))
            digest = sha256_file(packed)
            lock = {'repository': REPO, 'douzero_commit': sha, 'archive_sha256': digest,
                    'upstream_files': files, 'doutpu_sha256': local_hash,
                    'engine': 'douzero', 'encoding_schema': 1}
            copy_verified(packed, archive_path, digest)
            # Lock is the source archive's commit marker.
            atomic_bytes(lock_path, json_bytes(lock))
            safe_extract(packed, vendor)
    verify_upstream_files(vendor, lock)
    atomic_bytes(root/'source_lock.json', json_bytes(lock))
    print(json.dumps({'vendor': str(vendor), 'source_lock': str(lock_path),
                      'douzero_commit': lock['douzero_commit']}, indent=2), flush=True)
    return lock


def verify_upstream_files(vendor, lock):
    vendor = Path(vendor)
    if vendor.is_symlink():
        raise RuntimeError('Upstream source directory may not be a symlink')
    vendor = vendor.resolve()
    for relative, digest in lock['upstream_files'].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise RuntimeError('Extracted source hash/path mismatch: '+relative)
        # Check each raw component: resolve() would hide both file symlinks
        # and a symlinked package directory before the source check sees them.
        source = vendor
        for part in relative_path.parts:
            source = source/part
            if source.is_symlink():
                raise RuntimeError('Upstream source may not contain symlinks: '+relative)
        source = source.resolve()
        if vendor not in source.parents or sha256_file(source) != digest:
            raise RuntimeError('Extracted source hash/path mismatch: '+relative)
    # A leftover package can shadow a correctly hashed module (utils/ versus
    # utils.py). Cached bytecode is normal; standalone bytecode/native modules
    # are additional importable code and must not bypass this source manifest.
    importable = {'.py', '.pyc', '.pyo', '.so', '.pyd'}
    expected = {str(Path(name)) for name in lock['upstream_files']
                if Path(name).parts[0] == 'douzero' and Path(name).suffix in importable}
    actual = set()
    package = vendor/'douzero'
    for source in [package, *package.rglob('*')]:
        if source.is_symlink():
            raise RuntimeError('Upstream source may not contain symlinks: '+str(source.relative_to(vendor)))
        if '__pycache__' in source.relative_to(vendor).parts:
            continue
        if source.is_file() and source.suffix in importable:
            actual.add(str(source.relative_to(vendor)))
    if actual != expected:
        raise RuntimeError('Untracked or missing importable upstream source: '
                           + ', '.join(sorted(actual ^ expected)))


def verify_local_source(root, lock):
    root = Path(root).resolve()
    if lock.get('engine') != 'douzero' or lock.get('encoding_schema') != 1:
        raise RuntimeError('Unexpected rule engine or encoding schema')
    if lock.get('doutpu_sha256') != project_hash(root):
        raise RuntimeError('Trainer source differs from the run lock; prepare a new run and import the checkpoint.')
    if lock.get('douzero_commit') != PINNED_COMMIT:
        raise RuntimeError('Unexpected DouZero rule commit')
    verify_upstream_files(root/'vendor', lock)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=str(Path(__file__).resolve().parent))
    ap.add_argument('--savedir', required=True)
    ap.add_argument('--commit', default=PINNED_COMMIT, help='Full rule-engine commit SHA')
    ap.add_argument('--upstream-cache', default='')
    args = ap.parse_args()
    prepare(args.root, args.savedir, args.commit, args.upstream_cache or None)
