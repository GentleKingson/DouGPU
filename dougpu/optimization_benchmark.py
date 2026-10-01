"""Screen execution candidates on one verified checkpoint and real game requests.

This is a synchronized microbenchmark, not end-to-end throughput or a quality test.
No checkpoint, optimizer, replay or training RNG is modified.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
from .config import ModelConfig, TrainConfig
from .checkpoint import Store, atomic_bytes, json_bytes, sha256_file
from .runtime import configure_runtime, verify_backend


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--baseline-source', default='')
    ap.add_argument('--waves', type=int, default=80)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20260928)
    a = ap.parse_args()
    if a.waves < 1 or a.repeats < 1:
        ap.error('waves and repeats must be positive')
    spec = json.loads(Path(a.config).read_text())
    mc = ModelConfig(**spec['model']).validate()
    tc = TrainConfig(**spec['train']).validate()
    configure_runtime(tc)
    import jax
    from .actors import Ring
    from .encoding import pack_requests
    from .inference import Inference, merge_inference_stats
    from .model import make_train_step
    from .replay import Replay, group_history
    verify_backend(tc, jax)
    if tc.attention_impl != 'manual':
        raise ValueError('This inherited microbenchmark compares manual attention only. Use tune_local.py for attention changes.')
    saved = Store(a.checkpoint_dir).load_latest()
    if saved is None or saved['meta']['model'] != asdict(mc):
        raise ValueError('A verified checkpoint with the same model is required')
    if saved['meta']['train']['engine'] != tc.engine:
        raise ValueError('Checkpoint and benchmark rule engines differ')
    baseline_class = Inference
    if a.baseline_source:
        path = Path(a.baseline_source)/'dougpu'/'inference.py'
        if not path.is_file():
            path = Path(a.baseline_source)/'doutpu'/'inference.py'
        module = importlib.util.spec_from_file_location('dougpu._baseline_inference', path)
        baseline_module = importlib.util.module_from_spec(module)
        module.loader.exec_module(baseline_module)
        baseline_class = baseline_module.Inference
    p, opt = jax.device_put((saved['params'], saved['optimizer']))
    baseline = baseline_class(mc, tc.infer_batch, tc.action_chunk)
    ring = Ring(tc.engine, tc.infer_batch, a.seed)
    rng = np.random.default_rng(a.seed+1)
    corpus, expected = [], []
    digest = hashlib.sha256()
    for tick in range(64+a.waves):
        req = ring.requests()
        if tick >= 64:
            corpus.append(req)
            expected.append(baseline.choose(p, req))
            for value in pack_requests(req).values():
                digest.update(value.tobytes())
        choices = baseline.choose(p, req, rng, tc.epsilon_end)
        ring.advance(choices, 0)
    result = {'backend': jax.default_backend(), 'jax': jax.__version__,
              'device_kind': jax.devices()[0].device_kind, 'model': asdict(mc),
              'checkpoint_sha256': sha256_file(saved['path']),
              'checkpoint_updates': saved['meta']['updates'], 'seed': a.seed,
              'requests': sum(map(len, corpus)), 'corpus_sha256': digest.hexdigest(),
              'note': 'Same-weight real-request screening; not a full-job or held-out quality claim',
              'inference': {}, 'learner': {}}
    candidates = {'baseline': baseline,
                  'history': Inference(mc, tc.infer_batch, tc.action_chunk, history_buckets=True),
                  'fused': Inference(mc, tc.infer_batch, tc.action_chunk, fused=True),
                  'history_fused': Inference(mc, tc.infer_batch, tc.action_chunk,
                                             history_buckets=True, fused=True)}
    padding_reference = Inference(mc, tc.infer_batch, tc.action_chunk)
    for name, infer in candidates.items():
        start = time.perf_counter()
        mismatch, stats = 0, []
        for req, ref in zip(corpus, expected):
            actual = infer.choose(p, req)
            mismatch += int(np.count_nonzero(np.asarray(actual) != ref))
            if name == 'baseline' and a.baseline_source:
                for _ in padding_reference._batches(req):
                    pass
                stats.append(dict(padding_reference.last_stats))
            else:
                stats.append(dict(infer.last_stats))
        rng_a, rng_b = np.random.default_rng(a.seed+2), np.random.default_rng(a.seed+2)
        for req in corpus[:4]:
            assert baseline.choose(p, req, rng_a, .4) == infer.choose(p, req, rng_b, .4) or mismatch
        rng_equal = rng_a.bit_generator.state == rng_b.bit_generator.state
        result['inference'][name] = {'action_mismatches': mismatch, 'rng_state_equal': rng_equal,
            'parity_pass': mismatch == 0 and rng_equal,
            'warmup_plus_parity_seconds': time.perf_counter()-start,
            'padding': merge_inference_stats(stats),
            'round_seconds': [], 'wave_seconds': []}
        print(name, 'mismatches:', mismatch, flush=True)
    # Alternate order after every candidate has seen the entire shape corpus.
    names = list(candidates)
    for repeat in range(a.repeats):
        for name in (names if repeat % 2 == 0 else names[::-1]):
            entry = result['inference'][name]
            start = time.perf_counter()
            for req in corpus:
                tick = time.perf_counter()
                candidates[name].choose(p, req)  # device_get inside choose is the sync boundary
                entry['wave_seconds'].append(time.perf_counter()-tick)
            entry['round_seconds'].append(time.perf_counter()-start)
    baseline_seconds = np.median(result['inference']['baseline']['round_seconds'])
    for entry in result['inference'].values():
        entry['median_round_seconds'] = float(np.median(entry['round_seconds']))
        entry['throughput_gain'] = float(baseline_seconds/entry['median_round_seconds']-1)
        entry['wave_p50_p90_p99_seconds'] = np.quantile(entry.pop('wave_seconds'), [.5,.9,.99]).tolist()
    atomic_bytes(a.output, json_bytes(result))
    print(json.dumps(result['inference'], indent=2), flush=True)

    if saved['replay'] is None:
        raise ValueError('Real replay is required for the learner benchmark')
    replay = Replay(tc.replay_capacity)
    replay.restore(saved['replay'])
    updates_unit = saved['meta']['train'].get('replay_version_updates', False)
    sample_tc = replace(tc, replay_version_updates=updates_unit)
    version = saved['meta']['updates' if updates_unit else 'cycle']
    sample_rng = np.random.default_rng(a.seed+3)
    batches = [replay.sample(sample_tc, sample_rng, version) for _ in range(3)]
    batches = [jax.device_put(group_history(b, tc.history_groups) if tc.history_groups > 1 else b)
               for b in batches]
    steps = {name: make_train_step(mc, replace(tc, learner_remat=remat))
             for name, remat in [('remat', True), ('no_remat', False)]}
    outputs = {}
    for name, step in steps.items():
        tick = time.perf_counter()
        outputs[name] = [jax.block_until_ready(step(p, opt, batch)) for batch in batches]
        result['learner'][name] = {'compile_plus_first_steps_seconds': time.perf_counter()-tick,
                                  'step_seconds': []}
    for repeat in range(20):
        for name in (list(steps) if repeat % 2 == 0 else list(steps)[::-1]):
            tick = time.perf_counter()
            output = jax.block_until_ready(steps[name](p, opt, batches[repeat % len(batches)]))
            assert np.asarray(output[2])[-1] == 1 and np.isfinite(np.asarray(output[2])).all()
            result['learner'][name]['step_seconds'].append(time.perf_counter()-tick)
    diffs, moment_diffs, loss_diffs = [], [], []
    for (pa, oa, ma), (pb, ob, mb) in zip(outputs['remat'], outputs['no_remat']):
        assert int(oa['step']) == int(ob['step']) == int(opt['step'])+1
        diffs.extend(float(np.max(np.abs(np.asarray(pa[k])-np.asarray(pb[k])))) for k in pa)
        moment_diffs.extend(float(np.max(np.abs(np.asarray(oa['m'][k])-np.asarray(ob['m'][k])))) for k in pa)
        loss_diffs.append(float(np.max(np.abs(np.asarray(ma)-np.asarray(mb)))))
    result['learner']['comparison'] = {'max_parameter_abs_difference': max(diffs),
        'max_adam_m_abs_difference': max(moment_diffs), 'max_metrics_abs_difference': max(loss_diffs)}
    for name in steps:
        result['learner'][name]['median_step_seconds'] = float(np.median(result['learner'][name]['step_seconds']))
    result['memory_stats'] = jax.devices()[0].memory_stats()
    atomic_bytes(a.output, json_bytes(result))
    print(json.dumps(result['learner'], indent=2), flush=True)


if __name__ == '__main__':
    main()
