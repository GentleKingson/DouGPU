"""Read-only historical policy opponents. Only current-policy actions train."""
from dataclasses import asdict
import numpy as np
from .checkpoint import load_policy, sha256_file
from .encoding import PackedRequests
from .semantics import expected_parameter_shapes


def load_opponents(paths, model_config):
    expected = {k: shape for k, shape in expected_parameter_shapes(model_config).items()
                if k != 'ntp' and not k.startswith('belief')}
    policies, hashes = [], []
    for path in paths:
        params, meta = load_policy(path)
        if meta['model'] != asdict(model_config) or set(params) != set(expected):
            raise ValueError('Historical opponent model mismatch: '+path)
        for key, shape in expected.items():
            value = params[key]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid historical opponent array: '+key)
        policies.append(params)
        hashes.append(sha256_file(path))
    return policies, hashes


def policy_ids(requests):
    return (requests.data['policy_id'] if isinstance(requests, PackedRequests) else
            np.asarray([r['policy_id'] for r in requests], np.int32))


def choose_with_opponents(infer, params, policies, requests, rng, epsilon):
    from .inference import merge_inference_stats
    ids = policy_ids(requests)
    if np.any(ids < 0) or np.any(ids > len(policies)):
        raise ValueError('Unknown historical opponent ID')
    choices, stats = np.zeros(len(requests), np.int32), []
    for index, policy in enumerate([params, *policies]):
        selected = np.flatnonzero(ids == index)
        if len(selected):
            # ponytail: subset rows use existing unpacked inference; optimize only if measured costly.
            rows = requests if len(selected) == len(requests) else [requests[int(i)] for i in selected]
            choices[selected] = infer.choose(policy, rows, rng if index == 0 else None,
                                             epsilon if index == 0 else 0.)
            stats.append(dict(infer.last_stats))
    infer.last_stats = merge_inference_stats(stats)
    return choices.tolist()
