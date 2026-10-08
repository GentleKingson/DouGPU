"""Verified, generation-based checkpoints. Never deserialize Python pickle.

A checkpoint is considered committed only after its .ok.json exists. Remote
storage can still fail; hashes detect partial writes and older generations stay
available. Active actor games are intentionally not serialized.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import time
import uuid
import zipfile
import numpy as np
from .files import atomic_bytes, sha256_file

SCHEMA = 1


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')


def npz_bytes(arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def read_npz(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def copy_verified(src, dest, expected=None):
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.' + uuid.uuid4().hex + '.partial')
    expected = expected or sha256_file(src)
    try:
        with open(src, 'rb') as a, open(tmp, 'wb') as b:
            shutil.copyfileobj(a, b, length=8*1024*1024)
            b.flush()
            os.fsync(b.fileno())
        if sha256_file(tmp) != expected:
            raise IOError('Checkpoint verification failed during copy')
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def policy_bytes(params, model_config, metadata=None):
    # Auxiliary-only parameters are deliberately omitted from deployment.
    out = {k: np.asarray(v) for k, v in params.items()
           if k != 'ntp' and not k.startswith('belief')}
    out['__meta__'] = np.frombuffer(json_bytes({'schema': SCHEMA, 'model': model_config,
                                                'metadata': metadata or {}}), np.uint8)
    return npz_bytes(out)


def load_policy(path):
    out = read_npz(Path(path).read_bytes())
    meta = json.loads(out.pop('__meta__').tobytes().decode('utf-8'))
    if meta['schema'] != SCHEMA:
        raise ValueError('Unknown policy schema')
    return out, meta


class Store:
    def __init__(self, local, remote=None, keep=3):
        self.local = Path(local)
        self.remote = Path(remote) if remote else None
        self.keep = keep
        self.local.mkdir(parents=True, exist_ok=True)
        self.last_remote_ok = None

    def _prune(self, root):
        commits = []
        for marker in root.glob('ckpt_*.ok.json'):
            try:
                doc = json.loads(marker.read_text())
                # Match load_latest: invalid markers must not break the next save.
                if not isinstance(doc['file'], str) or Path(doc['file']).name != doc['file']:
                    continue
                if (not isinstance(doc['created_ns'], int) or isinstance(doc['created_ns'], bool)
                        or doc['created_ns'] < 0 or not isinstance(doc['sha256'], str)):
                    continue
                # Only count verified archives as safe survivors when pruning.
                archive = root / doc['file']
                if archive.exists() and sha256_file(archive) == doc['sha256']:
                    commits.append((doc['created_ns'], marker, archive))
            except (ValueError, OSError, KeyError, TypeError):
                continue
        commits.sort(reverse=True)
        for _, marker, archive in commits[self.keep:]:
            marker.unlink(missing_ok=True)
            archive.unlink(missing_ok=True)

    def save(self, params, optimizer, champion, replay, metadata, log_path=None):
        now = time.time_ns()
        name = f"ckpt_{metadata['cycle']:09d}_{metadata['updates']:012d}_{now}.zip"
        opt = {'step': optimizer['step']}
        for group in ('m', 'v'):
            opt.update({group + '::' + k: v for k, v in optimizer[group].items()})
        files = {'params.npz': npz_bytes(params), 'optimizer.npz': npz_bytes(opt),
                 'champion.npz': npz_bytes(champion),
                 'meta.json': json_bytes(dict(metadata, schema=SCHEMA))}
        if replay is not None:
            files['replay.npz'] = npz_bytes(replay.export())
        if log_path and Path(log_path).exists():
            files['metrics.jsonl'] = Path(log_path).read_bytes()
        manifest = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
        files['manifest.json'] = json_bytes(manifest)
        archive = self.local / name
        tmp = archive.with_suffix('.partial')
        try:
            with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED) as z:
                for k, value in files.items():
                    # NPZ entries are already compressed; only compress the plain log.
                    z.writestr(k, value, compress_type=(zipfile.ZIP_DEFLATED if k == 'metrics.jsonl'
                                                       else zipfile.ZIP_STORED), compresslevel=1)
            with open(tmp, 'rb') as f:
                os.fsync(f.fileno())
            os.replace(tmp, archive)
        finally:
            tmp.unlink(missing_ok=True)
        digest = sha256_file(archive)
        marker = {'file': name, 'sha256': digest, 'created_ns': now,
                  'cycle': metadata['cycle'], 'updates': metadata['updates']}
        marker_name = name[:-4] + '.ok.json'
        atomic_bytes(self.local / marker_name, json_bytes(marker))
        selection_seeds = metadata.get('selection_seeds', [])
        if 'eval_seed' in metadata.get('train', {}):
            selection_seeds = sorted(set(selection_seeds) | {metadata['train']['eval_seed']})
        exports = {'latest_policy.npz': policy_bytes(params, metadata['model'],
                                                      {'cycle': metadata['cycle'], 'selection_seeds': selection_seeds,
                                                       'selection': 'latest_not_selected'}),
                   'best_policy.npz': policy_bytes(champion, metadata['model'],
                                                   {'champion_cycle': metadata['champion_cycle'],
                                                    'selection_seeds': selection_seeds, 'selection': 'champion'})}
        for export_name, value in exports.items():
            atomic_bytes(self.local / export_name, value)
        self.last_remote_ok = None
        if self.remote:
            try:
                self.remote.mkdir(parents=True, exist_ok=True)
                copy_verified(archive, self.remote / name, digest)
                # Commit marker last; no claim that a Drive FUSE rename is transactional.
                atomic_bytes(self.remote / marker_name, json_bytes(marker))
                for export_name in exports:
                    copy_verified(self.local / export_name, self.remote / export_name)
                if log_path and Path(log_path).exists():
                    copy_verified(log_path, self.remote / 'metrics.jsonl')
                self._prune(self.remote)
                self.last_remote_ok = True
            except Exception as exc:
                self.last_remote_ok = False
                print(f'[WARNING] Checkpoint mirror failed; local checkpoint is valid: {exc}', flush=True)
        self._prune(self.local)
        return archive

    def load_latest(self):
        candidates = []
        observed_marker = False
        for root in [self.local] + ([self.remote] if self.remote else []):
            if not root.exists():
                continue
            for marker in root.glob('ckpt_*.ok.json'):
                observed_marker = True
                try:
                    doc = json.loads(marker.read_text())
                    # Do not accept path traversal through a corrupted marker.
                    if not isinstance(doc['file'], str) or Path(doc['file']).name != doc['file']:
                        continue
                    # Invalid timestamps must not prevent an older valid
                    # generation from being considered during sorting.
                    if (not isinstance(doc['created_ns'], int) or isinstance(doc['created_ns'], bool)
                            or doc['created_ns'] < 0 or not isinstance(doc['sha256'], str)):
                        continue
                    candidates.append((doc['created_ns'], root / doc['file'], doc))
                except (ValueError, OSError, KeyError, TypeError):
                    continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, path, doc in candidates:
            try:
                if sha256_file(path) != doc['sha256']:
                    raise ValueError('Archive hash mismatch')
                with zipfile.ZipFile(path) as z:
                    manifest = json.loads(z.read('manifest.json'))
                    raw = {name: z.read(name) for name in manifest}
                for name, digest in manifest.items():
                    if hashlib.sha256(raw[name]).hexdigest() != digest:
                        raise ValueError('Entry hash mismatch: ' + name)
                meta = json.loads(raw['meta.json'])
                if meta['schema'] != SCHEMA:
                    raise ValueError('Incompatible checkpoint schema')
                opt = read_npz(raw['optimizer.npz'])
                optimizer = {'step': opt['step'], 'm': {}, 'v': {}}
                for key, value in opt.items():
                    if '::' in key:
                        group, name = key.split('::', 1)
                        optimizer[group][name] = value
                return {'params': read_npz(raw['params.npz']), 'optimizer': optimizer,
                        'champion': read_npz(raw['champion.npz']), 'meta': meta,
                        'replay': read_npz(raw['replay.npz']) if 'replay.npz' in raw else None,
                        'log': raw.get('metrics.jsonl', b''), 'path': str(path),
                        'sha256': doc['sha256']}
            except Exception as exc:
                print(f'[WARNING] Skip invalid checkpoint {path.name}: {exc}', flush=True)
        if observed_marker:
            raise RuntimeError('Checkpoint markers exist but no valid checkpoint remains; not restarting silently')
        return None
