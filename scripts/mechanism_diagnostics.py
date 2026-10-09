"""Read a frozen checkpoint on CPU; independent diagnostic RNG, no optimizer or actor calls."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dougpu.checkpoint import Store
from dougpu.config import ModelConfig, TrainConfig
from dougpu.replay import Replay
from dougpu.semantics import check_array_state
from scripts.protocol_gate import boundaries


def sample_consumption(rows):
    """Successful sample uses, not draws from failed/prefetched attempts; fail on gaps."""
    sessions = boundaries(rows)
    total = 0
    for session in sessions:
        own = [r for r in rows if r['session_id'] == session['session_id']]
        starts = [r for r in own if r['event'] == 'start']
        if len(starts) != 1:
            raise ValueError('Missing session batch size')
        batch = starts[0]['effective_batch']
        if type(batch) is not int or batch < 3:
            raise ValueError('Invalid effective batch')
        total += batch * sum(r['successful_steps'] for r in own if r['event'] == 'train')
    return total


def noise_summary(gradients):
    gradients = np.asarray(gradients, dtype=np.float64)
    mean_squared = float(np.sum(gradients.mean(axis=0)**2))
    variance_trace = float(np.var(gradients, axis=0, ddof=1).sum())
    return dict(mean_gradient_squared_norm=mean_squared, variance_trace=variance_trace,
                variance_over_mean_squared=variance_trace/mean_squared if mean_squared else None,
                minibatch_gradient_norms=np.linalg.norm(gradients, axis=1).tolist())


def diagnose(directory, expected_sha, output, gradients=False):
    saved = Store(directory).load_latest()
    if saved is None or saved['sha256'] != expected_sha:
        raise ValueError('Wrong frozen checkpoint / fallback')
    meta = saved['meta']
    mc, tc = ModelConfig(**meta['model']), TrainConfig(**meta['train'])
    check_array_state(saved, mc)
    replay = Replay(tc.replay_capacity)
    replay.restore(saved['replay'])
    clock = meta['updates'] if tc.replay_version_updates else meta['cycle']
    bins = replay.strata(tc, clock)
    roles = []
    for role in range(3):
        ids = np.flatnonzero(replay.data['role'][:replay.size] == role)
        lag = clock - replay.data['version'][ids]
        roles.append(dict(role=role, available=int(len(ids)), eligible=int(len(bins[role])),
            excluded_fraction=1-len(bins[role])/len(ids) if len(ids) else None,
            lag_p50_p90_p99=np.percentile(lag, [50, 90, 99]).tolist() if len(ids) else None))
    rows = [json.loads(line) for line in saved['log'].splitlines()]
    try:
        uses = sample_consumption(rows)
    except (KeyError, ValueError, TypeError):
        uses = 'UNKNOWN'
    train = [r for r in rows if r['event'] == 'train']
    result = dict(checkpoint_sha256=saved['sha256'], training='PAUSED', optimizer_steps=0, GPU_work=0,
        diagnostic_seed=20261009, replay=dict(size=replay.size, clock=clock,
            unit='successful_updates' if tc.replay_version_updates else 'cycles', roles=roles,
            max_age_in_clock_units=tc.replay_max_age*(tc.updates_per_cycle if tc.replay_version_updates else 1)),
        consumption=dict(successful_sample_uses=uses, completed_samples=meta['complete_samples'],
            completed_samples_per_successful_update=meta['complete_samples']/meta['updates'] if meta['updates'] else None,
            successful_sample_uses_per_completed_sample=uses/meta['complete_samples']
                if isinstance(uses, int) and meta['complete_samples'] else 'UNKNOWN'),
        nonfinite=dict(logged_skips=sum(r.get('nonfinite_steps', 0) for r in train),
            skip_locations=[dict(session_id=r['session_id'], cycle=r['cycle'], count=r['nonfinite_steps'])
                            for r in train if r.get('nonfinite_steps', 0)],
            grad_norm_cycle_averages=[r['grad_norm'] for r in train],
            loss_cycle_averages=[r['loss'] for r in train],
            precision_comparison='NOT_MEASURED: no comparable GPU cuDNN fixed-input measurements'),
        gradient_noise='NOT_MEASURED', limitations=[
            'Version differences are neither elapsed seconds nor policy KL.',
            'Replay statistics describe the endpoint; no causal association with instability is identified.',
            'Successful sample uses exclude failed updates and unused prefetch draws.',
            'Logged loss/norm are successful-cycle averages, not individual failed-gradient measurements.',
            'An exception before a train event may omit the last failed attempts from checkpoint logs.'])
    if gradients:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import jax
        import jax.numpy as jnp
        from dougpu.model import sample_losses
        if jax.default_backend() != 'cpu':
            raise RuntimeError('CPU required')
        # ponytail: four correlated probes are exploratory only; use independent larger probes for inference.
        probe = replace(tc, micro_batch=3, accumulation=1, history_groups=1, attention_impl='manual')
        cfg = replace(mc, remat=tc.learner_remat) if tc.learner_remat is not None else mc
        value_grad = jax.jit(jax.value_and_grad(lambda p, b: sample_losses(p, b, cfg, probe), has_aux=True))
        params = jax.device_put(saved['params'])
        rng = np.random.default_rng(result['diagnostic_seed'])
        all_gradients = {key: [] for key in ('global', 'role0', 'role1', 'role2')}
        input_arrays, draw_ids, losses = {}, [], {key: [] for key in all_gradients}
        for index in range(4):
            # Copy RNG to record the same physical draws without consuming the probe's sampling stream.
            from copy import deepcopy
            recorder = deepcopy(rng)
            ids = [int(recorder.choice(bucket, 1, replace=True)[0]) for bucket in bins]
            recorder.shuffle(ids)
            draw_ids.extend(ids)
            batch = {k: v[0] for k, v in replay.sample(probe, rng, clock, bins).items()}
            assert np.array_equal(batch['role'], replay.data['role'][ids])
            input_arrays.update({f'{index}.{k}': v for k, v in batch.items()})
            for key in all_gradients:
                b = dict(batch)
                if key != 'global':
                    b['weight'] = b['weight']*(b['role'] == int(key[-1]))*3
                (loss, _), g = value_grad(params, jax.tree_util.tree_map(jnp.asarray, b))
                flat = np.concatenate([np.asarray(g[k]).ravel() for k in sorted(g)])
                if not np.isfinite(flat).all() or not np.isfinite(float(loss)):
                    raise ValueError('Nonfinite diagnostic gradient')
                all_gradients[key].append(flat)
                losses[key].append(float(loss))
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/'gradient-inputs.npz', **input_arrays, physical_indices=np.array(draw_ids))
        summaries = {k: dict(noise_summary(v), losses=losses[k]) for k, v in all_gradients.items()}
        result['gradient_noise'] = dict(status='MEASURED_EXPLORATORY', backend='cpu', jax=jax.__version__,
            minibatches=4, global_batch_size=3, role_batch_size=1, sampling_with_replacement=True,
            physical_indices=draw_ids, shared_replay_elements=len(draw_ids)-len(set(draw_ids)),
            repeated_draw_fraction=1-len(set(draw_ids))/len(draw_ids), precision_bf16=mc.bf16,
            attention_impl='manual', summaries=summaries,
            confidence='No calibrated CI: four small probes from one frozen replay; no optimal batch claim.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint_directory', type=Path)
    parser.add_argument('sha256')
    parser.add_argument('output', type=Path)
    parser.add_argument('--gradients', action='store_true')
    args = parser.parse_args()
    result = diagnose(args.checkpoint_directory, args.sha256, args.output, args.gradients)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'diagnostics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'nonfinite'}, indent=2))


if __name__ == '__main__':
    main()
