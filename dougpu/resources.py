"""Conservative CPU profiles, bounded by visible affinity and cgroup-v2 ancestors."""
import os
from pathlib import Path


def cpu_profile(cgroup_root='/sys/fs/cgroup', membership='/proc/self/cgroup', affinity=None):
    affinity = affinity or (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                            else (os.cpu_count() or 1))
    budget = float(affinity)
    root = Path(cgroup_root).resolve()
    current = root
    try:
        for row in Path(membership).read_text().splitlines():
            _, controllers, relative = row.split(':', 2)
            if not controllers:
                candidate = (root/relative.lstrip('/')).resolve()
                if candidate.is_dir() and (candidate == root or root in candidate.parents):
                    current = candidate
                break
    except (OSError, ValueError):
        pass
    quotas = []
    while current == root or root in current.parents:
        try:
            quota, period = (current/'cpu.max').read_text().split()
            if quota != 'max':
                limit = int(quota)/int(period)
                if limit > 0:
                    budget = min(budget, limit)
                    quotas.append({'path': str(current/'cpu.max'), 'cpus': limit})
        except (OSError, ValueError, ZeroDivisionError):
            pass
        if current == root:
            break
        current = current.parent
    workers = next(w for w in (32, 16, 8, 4, 2, 1) if w <= max(1, budget))
    if budget <= 4:
        workers = min(workers, 2)
    games = 256 if workers >= 16 else (128 if workers >= 8 else 64)
    return {'affinity': affinity, 'cpu_budget': budget, 'visible_quotas': quotas,
            'workers': workers, 'envs_per_worker': games//workers, 'infer_batch': games,
            'note': 'Only visible cgroup ancestors are inspectable; shared-host contention is not measured.'}
