"""Explicit accelerator selection and local diagnostics; no JAX import at module load."""
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def thread_environment():
    # Called by the launcher before importing NumPy. Actors inherit these limits.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = os.environ.get('DOUGPU_NUM_THREADS', '1')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')


def requested_backend(config):
    return config.backend or ('tpu' if config.require_tpu else 'cpu')


def configure_runtime(config, cache_dir=None):
    """Must run before JAX initializes. An explicit config overrides inherited CPU mode."""
    thread_environment()
    backend = requested_backend(config)
    if backend == 'cuda' and sys.platform != 'linux':
        raise RuntimeError('JAX CUDA requires Linux. On Windows use WSL2 Ubuntu, not native Python.')
    os.environ['JAX_PLATFORMS'] = backend
    if cache_dir:
        cache = Path(os.environ.get('JAX_COMPILATION_CACHE_DIR', str(cache_dir))).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['JAX_COMPILATION_CACHE_DIR'] = str(cache)
    if backend == 'cuda':
        # One owner, one visible GPU. This fraction is an allocator setting, not a hard VRAM cap.
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.70')
    return backend


def verify_backend(config, jax):
    requested = requested_backend(config)
    try:
        actual = jax.default_backend()
        devices = jax.devices()
    except Exception as exc:
        raise RuntimeError(f'Unable to initialize the requested {requested} backend. '
                           'Check the matching JAX accelerator packages and device/driver visibility; '
                           'no CPU fallback will be used.') from exc
    expected = 'gpu' if requested == 'cuda' else requested
    if actual != expected or (requested in ('cuda', 'tpu') and len(devices) != 1):
        raise RuntimeError(f'Requested exactly one {requested} accelerator; got {actual}: {devices}. '
                           'Refusing a misleading fallback or multi-device run.')
    if requested == 'cuda' and devices[0].platform != 'gpu':
        raise RuntimeError('Selected device is not a GPU')
    return devices


def package_versions():
    names = ('jax', 'jaxlib', 'numpy', 'scipy', 'ml_dtypes', 'jax-cuda13-plugin',
             'jax-cuda13-pjrt', 'nvidia-cuda-runtime', 'nvidia-cuda-nvcc',
             'nvidia-cudnn-cu13', 'nvidia-cublas', 'nvidia-cusolver',
             'jax-cuda12-plugin', 'jax-cuda12-pjrt')
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def git_source(root):
    """Best-effort session provenance, never a training or integrity gate."""
    source = {'git_commit': 'unknown', 'git_dirty': 'unknown'}
    for key, args in (('root', ['rev-parse', '--show-toplevel']),
                      ('git_commit', ['rev-parse', 'HEAD']),
                      ('git_dirty', ['status', '--porcelain', '--untracked-files=normal'])):
        try:
            value = subprocess.run(['git', '-C', str(root), *args], check=True,
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            if key == 'root':
                if not value or Path(value).resolve() != Path(root).resolve():
                    return source
            else:
                source[key] = bool(value) if key == 'git_dirty' else value or 'unknown'
        except (OSError, subprocess.SubprocessError):
            if key == 'root':
                return source
    return source


def nvidia_info():
    try:
        command = ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total',
                   '--format=csv,noheader,nounits']
        value = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if value.returncode:
            return {'available': False, 'error': value.stderr.strip()[:1500]}
        columns = ('index', 'name', 'uuid', 'driver_version', 'memory_mib')
        return {'available': True, 'gpus': [dict(zip(columns, (v.strip() for v in row.split(','))))
                                         for row in value.stdout.splitlines() if row.strip()]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'error': str(exc)}


def cpu_topology():
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    result = []
    for cpu in allowed:
        root = Path('/sys/devices/system/cpu') / f'cpu{cpu}'
        row = {'cpu': cpu}
        for name, rel in (('core_id', 'topology/core_id'), ('core_type_raw', 'topology/core_type'),
                          ('max_khz', 'cpufreq/cpuinfo_max_freq')):
            try:
                row[name] = (root / rel).read_text().strip()
            except OSError:
                pass
        result.append(row)
    return result


def diagnostics(config, cache_dir=None):
    """A real BF16 kernel probe plus host facts, not a training performance result."""
    configure_runtime(config, cache_dir)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from .resources import cpu_profile
    devices = verify_backend(config, jax)
    a = jnp.full((256, 256), 0.125, jnp.bfloat16)
    probe = jax.jit(lambda x: (x @ x).astype(jnp.float32))
    begin = time.perf_counter()
    result = np.asarray(probe(a))
    if not np.isfinite(result).all() or not np.allclose(result, 4.):
        raise RuntimeError('BF16 matrix multiply probe failed')
    document = {'requested_backend': requested_backend(config), 'backend': jax.default_backend(),
                'devices': [str(d) for d in devices], 'device_kind': devices[0].device_kind,
                'python': platform.python_version(), 'platform': platform.platform(),
                'wsl': 'microsoft' in platform.release().lower(), 'packages': package_versions(),
                'cpu': cpu_profile(), 'cpu_topology': cpu_topology(), 'nvidia_smi': nvidia_info(),
                'configured_parallelism': {'workers': config.workers,
                    'envs_per_worker': config.envs_per_worker, 'infer_batch': config.infer_batch},
                'bf16_kernel_pass': True, 'probe_compile_and_execute_seconds': time.perf_counter()-begin,
                'jax_cache': os.environ.get('JAX_COMPILATION_CACHE_DIR'),
                'allocator_preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
                'allocator_fraction': os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
                'thread_limits': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                'note': 'Kernel compatibility probe only; no target training throughput is inferred.'}
    if config.workers > document['cpu']['cpu_budget']:
        document['worker_warning'] = 'Configured actor count exceeds the visible CPU budget; tune fewer actors.'
    if os.environ.get('LD_LIBRARY_PATH'):
        document['library_path_warning'] = 'LD_LIBRARY_PATH is set; it can override pip CUDA libraries.'
    return document


def source_identity(root, snapshot):
    """Check startup source bytes and loaded module paths against prepare's ZIP."""
    import hashlib
    import zipfile
    from .files import sha256_file

    root, snapshot = Path(root).resolve(), Path(snapshot).resolve()
    paths = sorted([*root.glob('*.py'), *(root/'dougpu').glob('*.py')])
    files = {str(p.relative_to(root)): {'path': str(p.resolve()), 'sha256': sha256_file(p)}
             for p in paths}
    loaded = {}
    for name, module in list(sys.modules.items()):
        if (name == 'dougpu' or name.startswith('dougpu.') or name == 'bootstrap'
                or (name == '__main__' and getattr(module, '__file__', None)
                    and Path(module.__file__).resolve() == root/'dougpu/train.py')):
            path = Path(module.__file__).resolve()
            key = str(path.relative_to(root))
            if key not in files or files[key]['path'] != str(path):
                raise ValueError('Loaded source outside prepared files: ' + name)
            loaded[name] = key
    evidence = {'files': files, 'loaded_modules': loaded, 'snapshot_path': str(snapshot),
                'snapshot_sha256': None}
    if snapshot.exists():
        with zipfile.ZipFile(snapshot) as archive:
            names = archive.namelist()
            expected = {n for n in names if n.endswith('.py') and
                        (len(Path(n).parts) == 1 or (n.startswith('dougpu/') and len(Path(n).parts) == 2))}
            if len(names) != len(set(names)) or expected != set(files):
                raise ValueError('Prepared source file set mismatch')
            for name, record in files.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != record['sha256']:
                    raise ValueError('Prepared source mismatch: ' + name)
        evidence['snapshot_sha256'] = sha256_file(snapshot)
    return evidence
