"""Entry point: python -m dougpu.train --config ... --workdir ... --savedir ...

Only main() imports JAX. Spawned actors import this module without initializing
accelerators. A single foreground process exclusively owns the selected accelerator.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import signal
import time
import traceback
import uuid
import numpy as np
from .config import ModelConfig, TrainConfig
from .checkpoint import Store, atomic_bytes, json_bytes
from .replay import Replay, ReplayPrefetch, group_history
from .actors import ActorPool
from .runtime import configure_runtime, verify_backend, package_versions, git_source, source_identity
from .semantics import check_training_semantics, check_array_state


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--workdir', required=True)
    p.add_argument('--savedir', default='')
    p.add_argument('--source-lock', default='')
    return p.parse_args()


def main():
    args = parse_args()
    spec = json.loads(Path(args.config).read_text())
    mc, tc = ModelConfig(**spec.get('model', {})).validate(), TrainConfig(**spec.get('train', {})).validate()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    # Directory stays local to avoid thousands of small Drive FUSE writes.
    configure_runtime(tc, work/'jax_cache')
    import jax
    from .model import init_params, init_optimizer, make_train_step
    from .inference import Inference, merge_inference_stats
    from .evaluation import paired_evaluate, paired_difference, promotion_decision
    from .opponents import load_opponents, choose_with_opponents, policy_ids
    devices = verify_backend(tc, jax)
    print('[BACKEND]', jax.__version__, [str(d) for d in devices], flush=True)
    actual_lock = json.loads(Path(args.source_lock).read_text()) if args.source_lock else {'engine': tc.engine}
    if tc.engine == 'douzero' and 'douzero_commit' not in actual_lock:
        raise RuntimeError('Use bootstrap.py and supply --source-lock to pin the DouZero rule engine')
    if tc.engine == 'douzero':
        from bootstrap import verify_local_source
        verify_local_source(Path(__file__).resolve().parents[1], actual_lock)
    runtime_source = source_identity(Path(__file__).resolve().parents[1], work/'source'/'trainer_source.zip')
    store = Store(work / 'checkpoints', args.savedir or None, tc.keep_checkpoints)
    saved = store.load_latest() if tc.resume else None
    historical, historical_hashes = load_opponents(tc.historical_opponents, mc)
    historical = [jax.device_put(p) for p in historical]
    log_path = work / 'metrics.jsonl'
    rng = np.random.default_rng(tc.seed)
    opt = None
    replay = Replay(tc.replay_capacity)
    cycle = updates = frames = games = complete_samples = champion_cycle = 0
    prior_seconds = 0.
    collection_credit = 0.
    selection_seeds = {tc.eval_seed}
    historical_frames = np.zeros(len(historical), np.int64)
    if saved:
        meta = saved['meta']
        selection_seeds.update(meta.get('selection_seeds', [meta['train']['eval_seed']]))
        if meta['model'] != asdict(mc) or meta['train']['engine'] != tc.engine:
            raise ValueError('Model/engine mismatch on resume; use a NEW experiment directory')
        if meta.get('source_lock') != actual_lock:
            raise ValueError('Source lock mismatch on resume; do not mix rule-engine versions')
        # Operational settings may change; silently changing the objective may not.
        check_training_semantics(meta['train'], tc)
        if tc.target_updates is not None and tc.target_updates < meta['updates']:
            raise ValueError('target_updates is below the restored successful-update count')
        if meta.get('algorithm_experiment') and tc.target_updates != meta['algorithm_experiment']['target_updates']:
            raise ValueError('Cannot change the registered algorithm experiment endpoint on resume')
        if meta.get('historical_opponent_hashes', []) != historical_hashes:
            raise ValueError('Historical opponent bytes/order changed; use a new experiment')
        historical_frames[:] = meta.get('historical_frames', historical_frames)
        check_array_state(saved, mc)
        if meta['train'].get('sample_credit', False) != tc.sample_credit:
            raise ValueError('Sample credit semantics changed; use a reviewed migration into a new run.')
        if meta.get('versions', {}).get('jax') != jax.__version__:
            print('[WARNING] JAX version changed since checkpoint; numerical continuation is not bit-exact.', flush=True)
        params, opt, champion = (jax.device_put(saved[k]) for k in ('params', 'optimizer', 'champion'))
        if saved['replay'] is not None:
            replay.restore(saved['replay'])
        rng.bit_generator.state = meta['numpy_rng']
        cycle, updates, frames, games, complete_samples, champion_cycle = (
            meta[k] for k in ('cycle', 'updates', 'frames', 'games', 'complete_samples', 'champion_cycle'))
        prior_seconds = meta.get('total_seconds', 0.)
        if meta['train'].get('replay_version_updates', False) != tc.replay_version_updates:
            raise ValueError('Replay version unit changed; use a verified performance fork')
        collection_credit = meta.get('collection_credit', 0.) if tc.sample_credit else 0.
        atomic_bytes(log_path, saved['log'])
        print(f"[RESUME] {saved['path']} cycle={cycle}, updates={updates}, replay={replay.size}", flush=True)
        print('[RESUME] In-flight games restart; learner/replay/RNG restored, not bit-exact actor continuation.', flush=True)
    else:
        if not tc.resume and store.load_latest() is not None:
            raise ValueError('Existing experiment found; choose a new directory instead of overwriting it')
        params = init_params(mc, tc.seed)
        champion = params
        atomic_bytes(log_path, b'')
    opt = init_optimizer(params) if opt is None else opt
    infer = Inference(mc, tc.infer_batch, tc.action_chunk,
                      history_buckets=tc.infer_history_buckets, fused=tc.infer_fused,
                      attention_impl=tc.attention_impl)
    if tc.selfplay_kv_cache:
        from .kv_inference import KVInference
        infer = KVInference(mc, tc.workers*tc.envs_per_worker, tc.infer_batch,
                            tc.action_chunk, bucket_batch=False)
    eval_infer = Inference(mc, min(tc.infer_batch, 64), tc.action_chunk,
                           bucket_batch=tc.eval_bucket_batch,
                           history_buckets=tc.eval_optimized_inference and tc.infer_history_buckets,
                           fused=tc.eval_optimized_inference and tc.infer_fused,
                           attention_impl=tc.attention_impl)
    train_step = make_train_step(mc, tc)
    def prepare_batch(batch):
        if tc.attention_impl == 'cudnn':
            from .inference import validate_attention_tokens
            validate_attention_tokens(batch['tokens'], batch['lengths'])
        return jax.device_put(group_history(batch, tc.history_groups) if tc.history_groups > 1 else batch)
    atomic_bytes(work / 'resolved_config.json', json_bytes(spec))
    versions = {'python': platform.python_version(), 'jax': jax.__version__, 'numpy': np.__version__,
                'devices': [str(d) for d in devices], 'backend': jax.default_backend(),
                'packages': package_versions(), **git_source(Path(__file__).resolve().parents[1])}
    actor_start = None
    session_id = uuid.uuid4().hex
    resume_input = ({'path': saved['path'], 'sha256': saved['sha256'],
                     'updates': updates, 'cycle': cycle} if saved else None)
    start, last_save = time.monotonic(), 0.
    checkpoint_seconds_total = 0.
    stop = {'requested': False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))

    def stop_reason():
        if tc.target_updates is not None and updates >= tc.target_updates:
            return 'target_updates'
        if stop['requested']:
            return 'signal'
        if time.monotonic()-start >= tc.max_hours*3600:
            return 'time_limit'
        return None

    def stopped():
        return stop_reason() is not None

    def endpoint():
        return {'target_updates': tc.target_updates,
                'status': ('COMPLETE' if error is None and updates == tc.target_updates else 'INCOMPLETE'),
                'stop_reason': ('error' if error is not None else stop_reason() or
                                ('cycle_limit' if cycle >= tc.max_cycles else None))}

    def log(event):
        event = dict(event, session_id=session_id, wall_time=time.time(), cycle=cycle, updates=updates, frames=frames,
                     games=games, complete_samples=complete_samples,
                     total_seconds=prior_seconds + time.monotonic()-start)
        text = json.dumps(event, ensure_ascii=False, allow_nan=False)
        with open(log_path, 'a') as f:
            f.write(text+'\n')
        if event.get("event") not in ("train", "cycle_end") or cycle <= 3 or cycle % tc.log_every == 0:
            print(text, flush=True)

    def save(reason):
        nonlocal last_save, checkpoint_seconds_total
        save_began = time.monotonic()
        meta = {'model': asdict(mc), 'train': asdict(tc), 'source_lock': actual_lock,
                'versions': versions, 'numpy_rng': rng.bit_generator.state,
                'session_id': session_id, 'resume_input': resume_input,
                'runtime_source': runtime_source, 'actor_start': actor_start,
                'cycle': cycle, 'updates': updates, 'frames': frames, 'games': games,
                'complete_samples': complete_samples, 'champion_cycle': champion_cycle,
                'selection_seeds': sorted(selection_seeds),
                'historical_opponent_hashes': historical_hashes, 'historical_frames': historical_frames.tolist(),
                'collection_credit': collection_credit,
                'total_seconds': prior_seconds + time.monotonic()-start, 'reason': reason,
                'resume_scope': 'learner+optimizer+replay+main_rng; in-flight actor games restart'}
        if saved and 'algorithm_experiment' in saved['meta']:
            meta['algorithm_experiment'] = saved['meta']['algorithm_experiment']
        if tc.target_updates is not None:
            meta['update_endpoint'] = endpoint()
        path = store.save(params, opt, champion, replay if tc.save_replay else None, meta, log_path)
        last_save = time.monotonic()
        checkpoint_seconds_total += last_save-save_began
        print(f'[SAVE] {path.name} remote_ok={store.last_remote_ok} reason={reason}', flush=True)
        log({'event': 'checkpoint', 'reason': reason, 'remote_ok': store.last_remote_ok,
             'checkpoint_seconds': last_save-save_began})

    pool = None
    champion_rule = None
    error = None

    def account_samples(samples, winners):
        nonlocal fresh, collection_credit, complete_samples, games
        fresh += len(samples)
        if tc.sample_credit:
            collection_credit += len(samples)
        complete_samples += len(samples)
        games += len(winners)

    try:
        # A valid starting checkpoint exists even if the first JIT later fails.
        save('session_start')
        if args.savedir and store.last_remote_ok is not True:
            raise IOError('Initial checkpoint mirror failed; check the destination before a long session')
        if cycle < tc.max_cycles and not stopped():
            seeds = rng.integers(0, 2**32-1, tc.workers, dtype=np.uint64)
            if tc.ready_first:
                from .ready_actors import ReadyActorPool
                pool = ReadyActorPool(tc, seeds)
            else:
                pool = ActorPool(tc, seeds, packed=not tc.selfplay_kv_cache)
            actor_start = {'worker_seeds': seeds.tolist(), 'worker_order': list(range(tc.workers)),
                           'mode': 'ready_first' if tc.ready_first else 'ordered',
                           'packed': not (tc.ready_first or tc.selfplay_kv_cache), 'updates': updates,
                           'session_id': session_id}
            log({'event': 'actor_start', **actor_start})
        log({'event': 'start', 'parameters': sum(v.size for v in params.values()),
             'effective_batch': tc.batch_size, 'replay_bytes': sum(v.nbytes for v in replay.data.values()),
             'source_lock': actual_lock, 'versions': versions, 'resume_input': resume_input,
             'model': asdict(mc), 'train': asdict(tc), 'runtime_source': runtime_source})
        if historical:
            log({'event': 'historical_opponents', 'sha256': historical_hashes,
                 'episode_fraction': tc.historical_fraction, 'learner_landlord_probability': .5,
                 'replay': 'current-policy actions only', 'epsilon_clock': 'current-policy decisions only'})
        skipped = 0
        while cycle < tc.max_cycles and not stopped():
            began = time.monotonic()
            checkpoint_at_start = checkpoint_seconds_total
            evaluation_seconds = 0.
            updates_at_start, frames_at_start = updates, frames
            fresh, inference_seconds, actor_seconds = 0, 0., 0.
            replay_write_seconds = sample_seconds = 0.
            history_buckets = {}
            learner_shapes = {}
            nonfinite_steps_cycle = 0
            epsilon = tc.epsilon_start + min(frames/tc.epsilon_frames, 1.)*(tc.epsilon_end-tc.epsilon_start)
            last_stats = {}
            cycle_inference_stats = []
            decisions_by_policy = np.zeros(len(historical)+1, np.int64)
            while (collection_credit if tc.sample_credit else fresh) < tc.fresh_samples and not stopped():
                epsilon = tc.epsilon_start + min(frames/tc.epsilon_frames, 1.)*(tc.epsilon_end-tc.epsilon_start)
                if tc.ready_first:
                    tick = time.monotonic()
                    samples, winners = pool.collect()
                    actor_seconds += time.monotonic()-tick
                    replay.add(samples)
                    account_samples(samples, winners)
                    if collection_credit >= tc.fresh_samples:
                        break
                    selected = pool.select()
                    if not selected:
                        tick = time.monotonic()
                        samples, winners = pool.collect(block=True)
                        actor_seconds += time.monotonic()-tick
                        replay.add(samples)
                        account_samples(samples, winners)
                        continue
                    keys = [i*tc.envs_per_worker+j for i in selected for j in range(tc.envs_per_worker)]
                    tick = time.monotonic()
                    cache_args = {'cache_keys': keys, 'policy_version': updates} if tc.selfplay_kv_cache else {}
                    choices = infer.choose(params, [pool.requests[i] for i in keys], rng, epsilon, **cache_args)
                    inference_seconds += time.monotonic()-tick
                    last_stats = dict(infer.last_stats)
                    cycle_inference_stats.append(last_stats)
                    tick = time.monotonic()
                    pool.submit_selected(selected, choices, updates)
                    actor_seconds += time.monotonic()-tick
                    frames += len(keys)
                    if time.monotonic()-last_save >= tc.checkpoint_seconds:
                        save('periodic_collection')
                    continue
                group_stats = []
                count = len(pool.requests)
                learner_count = int(np.sum(policy_ids(pool.requests) == 0)) if historical else count
                workers_per_group = tc.workers // tc.actor_groups
                for group in range(tc.actor_groups):
                    first = group * workers_per_group
                    last = first + workers_per_group
                    tick = time.monotonic()
                    cache_args = {'cache_keys': range(first*tc.envs_per_worker, last*tc.envs_per_worker),
                                  'policy_version': updates} if tc.selfplay_kv_cache else {}
                    requests = pool.requests[first*tc.envs_per_worker:last*tc.envs_per_worker]
                    if historical:
                        counts = np.bincount(policy_ids(requests), minlength=len(historical)+1)
                        decisions_by_policy += counts
                        historical_frames += counts[1:]
                        choices = choose_with_opponents(infer, params, historical, requests, rng, epsilon)
                    else:
                        choices = infer.choose(params, requests, rng, epsilon, **cache_args)
                        decisions_by_policy[0] += len(requests)
                    inference_seconds += time.monotonic()-tick
                    group_stats.append(dict(infer.last_stats))
                    cycle_inference_stats.append(group_stats[-1])
                    tick = time.monotonic()
                    pool.submit(choices, updates if tc.replay_version_updates else cycle, first, last)
                    actor_seconds += time.monotonic()-tick
                last_stats = group_stats[0] if tc.actor_groups == 1 else {'groups': group_stats}
                tick = time.monotonic()
                samples, winners = pool.receive()
                actor_seconds += time.monotonic()-tick
                tick = time.monotonic()
                replay.add(samples)
                replay_write_seconds += time.monotonic()-tick
                account_samples(samples, winners)
                frames += learner_count
                if time.monotonic()-last_save >= tc.checkpoint_seconds:
                    save('periodic_collection')
            if tc.ready_first and not tc.actor_learner_overlap:
                tick = time.monotonic()
                samples, winners = pool.drain()
                actor_seconds += time.monotonic()-tick
                replay.add(samples)
                account_samples(samples, winners)
            learned, learner_seconds, metric_sum = 0, 0., np.zeros(6)
            collection_seconds = time.monotonic()-began
            # Small max-hour sessions can stop before any complete episode; valid,
            # but this does not constitute training and is clearly visible in logs.
            replay_version = updates if tc.replay_version_updates else cycle
            tick = time.monotonic()
            # No replay writes occur until this learner phase has finished.
            strata = replay.strata(tc, replay_version)
            ready = all(len(bucket) for bucket in strata)
            replay_index_seconds = time.monotonic()-tick
            learning_began = time.monotonic()
            if ready and not stopped():
                prefetch = ReplayPrefetch(replay, tc, rng, replay_version, prepare_batch, strata) if tc.learner_prefetch else None
                if prefetch is not None:
                    prefetch.prepare()
                for index in range(tc.updates_per_cycle):
                    if stopped():
                        break
                    if prefetch is not None:
                        batch, length = prefetch.take()
                    else:
                        tick = time.monotonic()
                        batch = replay.sample(tc, rng, replay_version, strata)
                        sample_seconds += time.monotonic()-tick
                        length = batch['tokens'].shape[-1]
                        batch = prepare_batch(batch)
                    history_buckets[length] = history_buckets.get(length, 0) + 1
                    parts = batch if isinstance(batch, tuple) else (batch,)
                    signature = '|'.join('x'.join(map(str, part['tokens'].shape)) for part in parts)
                    learner_shapes[signature] = learner_shapes.get(signature, 0) + 1
                    tick = time.monotonic()
                    params, opt, metrics = train_step(params, opt, batch)
                    if prefetch is not None and index+1 < tc.updates_per_cycle and not stopped():
                        prefetch.prepare()
                    met = np.asarray(metrics)  # synchronization: timing includes device execution
                    learner_seconds += time.monotonic()-tick
                    if met[-1] != 1 or not np.all(np.isfinite(met)):
                        nonfinite_steps_cycle += 1
                        skipped += 1
                        print('[WARNING] Non-finite step skipped; parameters/optimizer not advanced', flush=True)
                        if skipped >= 3:
                            raise FloatingPointError('Three non-finite steps; stop for diagnosis')
                        continue
                    skipped = 0
                    updates += 1
                    learned += 1
                    if tc.sample_credit:
                        collection_credit -= tc.fresh_samples/tc.updates_per_cycle
                    metric_sum += met
                    if time.monotonic()-last_save >= tc.checkpoint_seconds:
                        save('periodic_learning')
                if prefetch is not None:
                    sample_seconds += prefetch.sample_seconds
            learning_seconds = time.monotonic()-learning_began
            if tc.ready_first and tc.actor_learner_overlap:
                tick = time.monotonic()
                samples, winners = pool.drain()
                actor_seconds += time.monotonic()-tick
                replay.add(samples)
                account_samples(samples, winners)
            cycle += 1
            avg = metric_sum/max(learned, 1)
            log({'event': 'train', 'fresh_samples': fresh, 'replay_size': replay.size,
                 'epsilon': float(epsilon) if fresh or frames else tc.epsilon_start,
                 'loss': float(avg[0]), 'q_loss': float(avg[1]), 'ntp_loss': float(avg[2]),
                 'belief_loss': float(avg[3]), 'grad_norm': float(avg[4]), 'successful_steps': learned,
                 'nonfinite_steps': nonfinite_steps_cycle,
                 'inference_seconds': inference_seconds, 'actor_seconds': actor_seconds,
                 'learner_seconds': learner_seconds, 'cycle_seconds': time.monotonic()-began,
                 'learning_seconds': learning_seconds,
                 'replay_index_seconds': replay_index_seconds,
                 'replay_write_seconds': replay_write_seconds, 'sample_seconds': sample_seconds,
                 'history_buckets': history_buckets,
                 'learner_shapes': learner_shapes,
                 # Sample uses per newly completed sample in this cycle (denominator floored at 1).
                 'sample_update_ratio': learned*tc.batch_size/max(fresh, 1),
                 'padding': last_stats,
                 'padding_cycle': merge_inference_stats(cycle_inference_stats)})
            if historical:
                log({'event': 'opponent_usage', 'decisions_by_policy': decisions_by_policy.tolist(),
                     'historical_frames': historical_frames.tolist()})
            if tc.ready_first:
                log({'event': 'scheduler', 'counts': dict(pool.counts),
                     'worker_steps': list(pool.steps), 'collection_credit': collection_credit,
                     'behavior_update_lag_limit': tc.updates_per_cycle})
            if tc.eval_every > 0 and cycle % tc.eval_every == 0 and not stopped():
                tick = time.monotonic()
                result = paired_evaluate(params, champion, eval_infer, tc.engine, tc.eval_deals,
                                         tc.eval_seed, stopped, actor_pool=pool, workers=tc.eval_workers,
                                         kv_cache=tc.eval_kv_cache)
                elapsed = time.monotonic()-tick
                evaluation_seconds += elapsed
                head_to_head_seconds = elapsed
                if result is not None:
                    rule_result = None
                    if champion_rule is None and not stopped():
                        tick = time.monotonic()
                        champion_rule = paired_evaluate(champion, 'rule', eval_infer, tc.engine,
                                                       tc.eval_deals, tc.eval_seed, stopped,
                                                       actor_pool=pool, workers=tc.eval_workers,
                                                       kv_cache=tc.eval_kv_cache)
                        elapsed = time.monotonic()-tick
                        evaluation_seconds += elapsed
                        if champion_rule is not None:
                            log({'event': 'evaluation', 'opponent': 'champion_rule_reference',
                                 'result': champion_rule, 'evaluation_seconds': elapsed})
                    if champion_rule is not None and not stopped():
                        tick = time.monotonic()
                        rule_result = paired_evaluate(params, 'rule', eval_infer, tc.engine,
                                                      tc.eval_deals, tc.eval_seed, stopped,
                                                      actor_pool=pool, workers=tc.eval_workers,
                                                      kv_cache=tc.eval_kv_cache)
                        elapsed = time.monotonic()-tick
                        evaluation_seconds += elapsed
                        if rule_result is not None:
                            log({'event': 'evaluation', 'opponent': 'simple_rule', 'result': rule_result,
                                 'evaluation_seconds': elapsed})
                    difference = paired_difference(rule_result, champion_rule) if rule_result is not None else None
                    decision = (promotion_decision(result, difference, tc.promotion_role_margin)
                                if difference is not None else
                                {'promoted': False, 'checks': {'validation_complete': False}})
                    log({'event': 'evaluation', 'opponent': 'frozen_champion', 'result': result,
                         'promoted': decision['promoted'], 'promotion': decision,
                         'rule_difference': difference, 'champion_cycle_before': champion_cycle,
                         'evaluation_seconds': head_to_head_seconds})
                    if decision['promoted']:
                        champion, champion_cycle = params, cycle
                        champion_rule = rule_result
                        save('champion_promotion')
            if time.monotonic()-last_save >= tc.checkpoint_seconds:
                save('periodic')
            # Legacy train timings overlap; this wall interval includes evaluation and saves.
            log({'event': 'cycle_end', 'full_cycle_seconds': time.monotonic()-began,
                 'stop_reason': stop_reason(),
                 'updates_planned': tc.updates_per_cycle,
                 'nonfinite_steps': nonfinite_steps_cycle,
                 'updates_completed': updates-updates_at_start,
                 'selfplay_frames': frames-frames_at_start,
                 'collection_seconds': collection_seconds, 'learning_seconds': learning_seconds,
                 'replay_index_seconds': replay_index_seconds,
                 'evaluation_seconds': evaluation_seconds,
                 'checkpoint_seconds': checkpoint_seconds_total-checkpoint_at_start})
    except BaseException as exc:
        error = exc
        traceback.print_exc()
    finally:
        try:
            if pool is not None:
                pool.close()
        except BaseException as exc:
            if error is None:
                error = exc
        try:
            if error is None:
                # This records termination, not a successful checkpoint commit.
                log({'event': 'session_end', 'runtime_seconds': time.monotonic()-start,
                     'stop_reason': stop_reason() or 'cycle_limit',
                     **({'update_endpoint': endpoint()} if tc.target_updates is not None else {})})
            save('error' if error else 'session_end')
        except BaseException as exc:
            print(f'[ERROR] Final checkpoint failed: {exc}', flush=True)
            if error is None:
                error = exc
    if error is not None:
        raise error
    print(f'[DONE] cycle={cycle} updates={updates} games={games}; latest exports: {store.local}', flush=True)
    if args.savedir:
        if store.last_remote_ok:
            print('[MIRROR] Latest generation verified: '+args.savedir, flush=True)
        else:
            raise RuntimeError('Training ended but the checkpoint mirror failed. The local checkpoint remains available.')


if __name__ == '__main__':
    main()
