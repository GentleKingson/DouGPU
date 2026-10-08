#!/usr/bin/env python3
"""Plan or run one sequential, checkpoint-paired DouGPU RTX 5070 tuning stage.

Planning is the default. Add --execute to spend the displayed time budget.
Stop the source run first. Every trial forks the SAME verified private snapshot;
the source run and its original checkpoints are never rewritten.

Example:
  python tune_local.py --source-run runs/baseline --config configs/rtx5070_balanced.json \
    --output runs/tune_workers --stage workers
  python tune_local.py --source-run runs/baseline --config configs/rtx5070_balanced.json \
    --output runs/tune_workers --stage workers --repeats 3 --minutes 10 --execute

Pass a selected candidate's JSON to the next stage. No stage automatically
launches the next, changes the learning objective, or falls back after an OOM.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time
import zipfile

from dougpu.metrics_summary import rank_trials, summarize_metrics
from dougpu.files import write_json, sha256_file as sha256


ROOT = Path(__file__).resolve().parent
STAGES = ("workers", "envs", "inference", "chunks", "learner", "history", "remat", "attention")
TUNABLE = {
    "workers", "envs_per_worker", "infer_batch", "actor_groups", "eval_workers",
    "infer_history_buckets", "infer_fused", "action_chunk", "micro_batch",
    "accumulation", "history_groups", "learner_remat", "attention_impl",
}


def _configuration_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_base(config: dict) -> None:
    train = config.get("train", {})
    if train.get("backend") != "cuda" or train.get("require_tpu", True):
        raise ValueError("Tuning requires an explicit backend='cuda', require_tpu=false profile")
    if train.get("engine", "douzero") != "douzero":
        raise ValueError("RTX throughput candidates must use the real DouZero engine")
    if train.get("micro_batch", 32) * train.get("accumulation", 8) != 256:
        raise ValueError("This migration benchmark preserves effective batch 256")
    if train.get("fresh_samples", 2048) != 2048 or train.get("updates_per_cycle", 4) != 4:
        raise ValueError("This migration benchmark preserves fresh_samples=2048 and updates_per_cycle=4")
    if not train.get("save_replay", True) or not train.get("resume", True):
        raise ValueError("Trials must preserve replay and resume the verified learning state")
    if train.get("eval_every", 25) <= 0 or train.get("checkpoint_seconds", 300) <= 0:
        raise ValueError("End-to-end tuning requires normal evaluation and periodic checkpoints")
    if train.get("selfplay_kv_cache", False):
        raise ValueError("The research KV-cache path is outside this local tuning matrix")


def make_candidates(config: dict, stage: str) -> list[dict]:
    """Small sequential stages; no cross-product of unrelated performance knobs."""
    _validate_base(config)
    if stage not in STAGES:
        raise ValueError("Unknown stage: " + stage)
    base = copy.deepcopy(config)
    train = base["train"]
    candidates = [{"name": "baseline", "config": base, "changes": {}, "diagnostic_only": False}]
    hashes = {_configuration_hash(base)}

    def add(name: str, changes: dict, diagnostic: bool = False) -> None:
        if set(changes) - TUNABLE:
            raise ValueError("Candidate attempted to change a nonperformance setting")
        candidate = copy.deepcopy(base)
        candidate["train"].update(changes)
        _validate_base(candidate)
        tc = candidate["train"]
        if tc.get("workers", 8) % tc.get("actor_groups", 2):
            raise ValueError("Actor groups must divide the worker count")
        if tc.get("accumulation", 8) % tc.get("history_groups", 1):
            return  # Explicitly omit invalid history-group / accumulation geometries.
        digest = _configuration_hash(candidate)
        if digest in hashes:
            return
        hashes.add(digest)
        candidates.append({"name": name, "config": candidate,
                           "changes": {key: {"from": train.get(key), "to": value}
                                       for key, value in changes.items() if train.get(key) != value},
                           "diagnostic_only": diagnostic})

    if stage == "workers":
        for workers in (6, 8, 10, 12):
            add(f"workers_{workers:02d}", {
                "workers": workers, "envs_per_worker": 8, "actor_groups": 2,
                "infer_batch": _power_of_two((workers // 2) * 8),
                "eval_workers": min(train.get("eval_workers", 0), workers),
            })
    elif stage == "envs":
        workers = train.get("workers", 8)
        if workers % 2:
            raise ValueError("The envs stage needs an even worker count for actor_groups=2")
        for envs in (4, 8, 16, 32):
            batch = _power_of_two((workers // 2) * envs)
            add(f"envs_{envs:02d}_infer_{batch}", {
                "envs_per_worker": envs, "actor_groups": 2, "infer_batch": batch,
            })
    elif stage == "inference":
        # Compare each inference switch against the current baseline. A selected
        # winner may become the baseline for a second pass; do not mix switches.
        for key, label in (("infer_history_buckets", "history_buckets"), ("infer_fused", "fused")):
            for enabled in (False, True):
                add(label + ("_on" if enabled else "_off"), {key: enabled})
    elif stage == "chunks":
        for chunk in (1024, 2048):
            add(f"action_chunk_{chunk}", {"action_chunk": chunk})
    elif stage == "learner":
        for micro, accumulation in ((16, 16), (32, 8), (64, 4), (128, 2)):
            add(f"micro_{micro}_accum_{accumulation}", {"micro_batch": micro, "accumulation": accumulation})
    elif stage == "history":
        for groups in (1, 2, 4):
            add(f"history_groups_{groups}", {"history_groups": groups})
    elif stage == "remat":
        for enabled in (True, False):
            add("remat_" + ("on" if enabled else "off"), {"learner_remat": enabled})
    elif stage == "attention":
        for implementation in ("manual", "cudnn", "xla"):
            add("attention_" + implementation, {"attention_impl": implementation}, implementation == "xla")
    return candidates


def verify_latest(checkpoint_dir: Path) -> dict:
    """Verify archives and every manifested entry, without loading arrays/JAX."""
    candidates = []
    for marker in checkpoint_dir.glob("ckpt_*.ok.json"):
        try:
            marker_bytes = marker.read_bytes()
            document = json.loads(marker_bytes)
            filename = document["file"]
            if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".zip"):
                continue
            if marker.name != filename[:-4] + ".ok.json":
                continue
            candidates.append((int(document["created_ns"]), marker, marker_bytes, document))
        except (ValueError, OSError, KeyError, TypeError):
            continue
    failures = []
    for _, marker, marker_bytes, document in sorted(candidates, key=lambda item: item[0], reverse=True):
        archive = checkpoint_dir / document["file"]
        try:
            if sha256(archive) != document["sha256"]:
                raise ValueError("archive hash mismatch")
            with zipfile.ZipFile(archive) as bundle:
                if len(bundle.namelist()) != len(set(bundle.namelist())):
                    raise ValueError("duplicate archive entries")
                manifest = json.loads(bundle.read("manifest.json"))
                required = {"params.npz", "optimizer.npz", "champion.npz", "meta.json", "replay.npz"}
                if not isinstance(manifest, dict) or not required.issubset(manifest):
                    raise ValueError("checkpoint lacks complete optimizer/replay state")
                for entry, expected in manifest.items():
                    with bundle.open(entry) as stream:
                        digest = hashlib.file_digest(stream, "sha256")
                    if digest.hexdigest() != expected:
                        raise ValueError("entry hash mismatch: " + entry)
                meta = json.loads(bundle.read("meta.json"))
            if meta.get("schema") != 1:
                raise ValueError("unsupported checkpoint schema")
            if meta.get("cycle") != document.get("cycle") or meta.get("updates") != document.get("updates"):
                raise ValueError("marker counters differ from checkpoint metadata")
            return {"archive": archive, "marker": marker, "marker_bytes": marker_bytes,
                    "document": document, "meta": meta, "skipped_invalid": failures}
        except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            failures.append({"archive": archive.name, "reason": str(exc)})
    raise ValueError("No complete verified checkpoint with replay in " + str(checkpoint_dir)
                     + (": " + json.dumps(failures) if failures else ""))


@contextmanager
def source_guard(source_run: Path):
    """Use the runner's existing lock without creating or editing source files."""
    lock = source_run / ".run.lock"
    if not lock.exists():
        print("[SOURCE] No runner lock exists; the source run must remain stopped.", flush=True)
        yield
        return
    import fcntl  # This GPU package targets Linux / WSL2.
    with lock.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Source run is active. Stop it before taking a tuning snapshot.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def snapshot_source(source_run: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError("Snapshot already exists; use a new output directory: " + str(destination))
    verified = verify_latest(source_run / "checkpoints")
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    archive = verified["archive"]
    target = checkpoint_dir / archive.name
    shutil.copyfile(archive, target)
    expected = verified["document"]["sha256"]
    if sha256(target) != expected or sha256(archive) != expected:
        raise ValueError("Source archive changed or snapshot copy failed verification")
    if verified["marker"].read_bytes() != verified["marker_bytes"]:
        raise ValueError("Source checkpoint marker changed during snapshot")
    # Commit marker is written only after the private archive has been verified.
    (checkpoint_dir / verified["marker"].name).write_bytes(verified["marker_bytes"])
    evidence = {
        "source_run": str(source_run), "source_archive": str(archive),
        "archive": str(target), "sha256": expected,
        "marker_sha256": hashlib.sha256(verified["marker_bytes"]).hexdigest(),
        "cycle": verified["meta"]["cycle"], "updates": verified["meta"]["updates"],
        "source_lock": verified["meta"].get("source_lock"),
        "skipped_invalid_source_checkpoints": verified["skipped_invalid"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "learner, optimizer, replay, champion and main RNG; in-flight games restart",
    }
    write_json(destination / "evidence.json", evidence)
    return evidence


def _run(command: list[str], log_path: Path, environment: dict) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    with log_path.open("ab") as log:
        log.write(("\n[COMMAND] " + json.dumps(command) + "\n").encode())
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        interrupted = {"signal": None}

        def forward(signum, _frame):
            interrupted["signal"] = signum
            if process.poll() is None:
                print("[STOP] Waiting for the runner to finish its safe checkpoint before leaving the stage.", flush=True)
                process.send_signal(signum)

        previous = {signum: signal.signal(signum, forward) for signum in (signal.SIGINT, signal.SIGTERM)}
        try:
            returncode = process.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        if interrupted["signal"] is not None:
            raise KeyboardInterrupt("Stage stopped after runner exit; inspect the current trial's checkpoint/log.")
    return returncode, time.monotonic() - began


def run_trial(args, candidate: dict, repeat: int, snapshot: dict, environment: dict) -> dict:
    run_dir = args.output / "trials" / f"repeat_{repeat:02d}" / candidate["name"]
    config_path = args.output / "configs" / (candidate["name"] + ".json")
    log_path = args.output / "logs" / f"repeat_{repeat:02d}_{candidate['name']}.log"
    steps = [
        ("prepare", [sys.executable, str(ROOT / "run_local.py"), "prepare", "--run-dir", str(run_dir),
                     "--config", str(config_path), "--upstream-cache", str(args.source_run / "source")]),
        ("import", [sys.executable, str(ROOT / "run_local.py"), "import-checkpoint", "--run-dir", str(run_dir),
                    "--source-state", str(args.output / "snapshot" / "checkpoints")]),
        ("preflight", [sys.executable, str(ROOT / "run_local.py"), "preflight", "--run-dir", str(run_dir)]),
        ("train", [sys.executable, str(ROOT / "run_local.py"), "train", "--run-dir", str(run_dir),
                   "--hours", str(args.minutes / 60), "--cycles", str(args.cycles)]),
    ]
    result = {"candidate": candidate["name"], "repeat": repeat, "stage": args.stage,
              "diagnostic_only": candidate["diagnostic_only"], "run_dir": str(run_dir),
              "config": str(config_path), "config_sha256": _configuration_hash(candidate["config"]),
              "source_snapshot_sha256": snapshot["sha256"],
              "commands": [{"step": step, "argv": command} for step, command in steps],
              "log": str(log_path), "returncode": None, "command_results": []}
    print(f"[TRIAL] repeat={repeat}/{args.repeats} candidate={candidate['name']} "
          f"training cap={args.minutes:g} min", flush=True)
    for step, command in steps:
        returncode, elapsed = _run(command, log_path, environment)
        result["command_results"].append({"step": step, "returncode": returncode, "wall_seconds": elapsed})
        result["returncode"] = returncode
        if returncode:
            result["failed_step"] = step
            break
    training_result = next((entry for entry in result["command_results"] if entry["step"] == "train"), None)
    try:
        if training_result is None:
            raise ValueError("Training did not start; inspect launcher log")
        summary = summarize_metrics(run_dir / "metrics.jsonl", config=candidate["config"],
                                    warmup_cycles=args.warmup_cycles, process_returncode=training_result["returncode"],
                                    process_wall_seconds=training_result["wall_seconds"])
        if result["returncode"] == 0:
            terminal = verify_latest(run_dir / "checkpoints")
            result["terminal_checkpoint"] = {"archive": str(terminal["archive"]),
                                               "sha256": terminal["document"]["sha256"],
                                               "reason": terminal["meta"].get("reason")}
            if terminal["meta"].get("reason") != "session_end":
                summary["eligible"] = False
                summary["reasons"].append("latest_verified_checkpoint_is_not_session_end")
            if terminal["skipped_invalid"]:
                summary["eligible"] = False
                summary["reasons"].append("newer_checkpoint_failed_verification; cannot_credit_an_older_terminal_save")
            for counter, value in summary.get("terminal_checkpoint_counters", {}).items():
                if terminal["meta"].get(counter) != value:
                    summary["eligible"] = False
                    summary["reasons"].append("terminal_checkpoint_counter_mismatch:" + counter)
        result["summary"] = summary
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["summary"] = {"eligible": False, "reasons": [str(exc)]}
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Non-finite step skipped" in log_text or "Three non-finite steps" in log_text:
        result["summary"]["eligible"] = False
        result["summary"]["reasons"].append("nonfinite_optimizer_step_in_launcher_log")
    if any(message in log_text for message in ("RESOURCE_EXHAUSTED", "CUDA_ERROR_OUT_OF_MEMORY", "out of memory")):
        result["failure_class"] = "out_of_memory_reported; no_configuration_fallback"
        result["summary"]["eligible"] = False
        result["summary"]["reasons"].append("out_of_memory_reported_in_launcher_log")
    write_json(args.output / "results" / f"repeat_{repeat:02d}_{candidate['name']}.json", result)
    print("[RESULT] " + candidate["name"] + " " +
          ("eligible" if result["summary"].get("eligible") else "ineligible") +
          "; exit=" + str(result["returncode"]), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-run", type=Path, required=True, help="Stopped prepared run with a verified checkpoint/replay")
    parser.add_argument("--config", type=Path, required=True, help="GPU baseline JSON; learned model/objectives remain fixed")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this one stage")
    parser.add_argument("--stage", choices=STAGES, default="workers")
    parser.add_argument("--minutes", type=float, default=10,
                        help="Training-loop cap per candidate/repeat; startup and shutdown add time (default: 10)")
    parser.add_argument("--cycles", type=int, default=1000000,
                        help="Additional cycle cap per trial; the time cap usually stops it first")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-cycles", type=int, default=25,
                        help="Cycles excluded after each NEW session starts, not global checkpoint cycles")
    parser.add_argument("--execute", action="store_true", help="Run the printed sequential plan; otherwise only write the plan")
    args = parser.parse_args()
    if args.minutes <= 0 or args.cycles < 1 or args.repeats < 1 or args.warmup_cycles < 0:
        parser.error("minutes/cycles/repeats must be positive; warmup-cycles must be nonnegative")
    args.source_run = args.source_run.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.source_run == args.output or args.source_run in args.output.parents or args.output in args.source_run.parents:
        parser.error("Source and output directories must be separate, with neither containing the other")
    if not (args.source_run / "source" / "source_lock.json").exists():
        parser.error("source-run must be prepared by run_local.py and contain source/source_lock.json")
    if (args.output / "snapshot").exists() or (args.output / "trials").exists():
        parser.error("Output already contains an executed experiment; use a new directory")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = make_candidates(config, args.stage)
    plan = {
        "schema": 1, "mode": "execute" if args.execute else "plan_only", "stage": args.stage,
        "source_run": str(args.source_run), "baseline_config": str(args.config),
        "repeats": args.repeats, "minutes_per_trial": args.minutes, "warmup_cycles": args.warmup_cycles,
        "estimated_training_budget_minutes": len(candidates) * args.repeats * args.minutes,
        "budget_note": "Training-loop caps only; preparation, checkpoint import, mandatory numerical/full-history preflight, JIT/import and final writes add wall time.",
        "candidate_count": len(candidates), "candidates": candidates,
        "invariants": {"effective_batch": 256, "fresh_samples": 2048, "updates_per_cycle": 4,
                       "objectives_model_replay": "unchanged", "evaluation_schedule": "unchanged",
                       "single_gpu_owner": "all prepare/import/train processes run sequentially"},
        "measurement": "Aggregate successful optimizer updates divided by complete full-cycle wall seconds; "
                       "includes scheduled evaluations and in-cycle saves. Learner-only rate is separate.",
        "limitations": [
            "Stop the source training process for the entire stage; do not modify its source cache.",
            "Each repeat restarts in-flight games from the same learner/optimizer/replay/main-RNG snapshot; actor trajectories differ across shapes/workers.",
            "A short budget may not cover 25 warmup cycles, another full evaluation interval and a periodic save. Such trials are unrankable; explicitly increase --minutes.",
            "Late new static shapes retain compilation time in the measured interval and are reported.",
            "Three sufficient paired repeats support only this tested stage; no global optimum or playing-strength claim.",
            "History groups that do not divide accumulation are omitted. Attention xla is diagnostic and excluded from winner selection.",
            "The workers/envs stages adjust inference capacity with actor geometry; inference and remat are separate experiments.",
        ],
        "next_stage": "Choose a tested candidate JSON explicitly as --config for another stage; no automatic chained long run.",
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR", str(args.output / "jax_cache")),
                        "cache_policy": "shared sequentially across candidates/repeats; preflight is run for every trial"},
    }
    write_json(args.output / "plan.json", plan)
    for candidate in candidates:
        write_json(args.output / "configs" / (candidate["name"] + ".json"), candidate["config"])
    print(json.dumps({key: plan[key] for key in ("mode", "stage", "candidate_count", "repeats",
                                                 "estimated_training_budget_minutes", "budget_note")}, indent=2), flush=True)
    print("[PLAN] " + str(args.output / "plan.json"), flush=True)
    if not args.execute:
        return
    environment = dict(os.environ)
    environment.setdefault("JAX_COMPILATION_CACHE_DIR", str(args.output / "jax_cache"))
    trials = []
    try:
        with source_guard(args.source_run):
            snapshot = snapshot_source(args.source_run, args.output / "snapshot")
            for repeat in range(1, args.repeats + 1):
                # Counterbalance order to reduce an always-first baseline thermal bias.
                order = list(candidates)
                if repeat % 2 == 0:
                    order.reverse()
                elif repeat > 1:
                    shift = (repeat - 1) % len(order)
                    order = order[shift:] + order[:shift]
                for candidate in order:
                    trials.append(run_trial(args, candidate, repeat, snapshot, environment))
                    report = rank_trials(trials, required_repeats=args.repeats)
                    report.update({"stage": args.stage, "snapshot": snapshot, "trials": trials})
                    write_json(args.output / "summary.json", report)
    except KeyboardInterrupt:
        write_json(args.output / "interrupted.json", {
            "status": "interrupted", "completed_trials": len(trials),
            "note": "Runner was allowed to finish its checkpoint. The interrupted trial is not ranked; inspect its log and checkpoints.",
        })
        raise SystemExit(130)
    print("[SUMMARY] " + str(args.output / "summary.json"), flush=True)
    if report["best_observed_candidate"]:
        print("[" + report["status"].upper() + "] best observed: " + report["best_observed_candidate"], flush=True)
    else:
        print("[UNRANKABLE] No sufficient full-cycle comparison; inspect reasons before increasing the budget.", flush=True)


if __name__ == "__main__":
    main()
