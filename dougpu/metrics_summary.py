"""Summarize complete, recent training cycles without importing JAX.

The primary rate is successful optimizer updates / aggregate full-cycle wall
time. Evaluation and in-cycle checkpoint writes are included in that time.
Startup, warmup, an incomplete time-limited tail, and the final checkpoint are
reported separately. A learner-only rate is deliberately labelled as such.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


PADDING_COUNTS = (
    "requests", "legal_actions", "action_slots", "history_tokens",
    "history_slots", "state_slots", "encoder_batches", "fused_batches",
)
EXPECTED_OPPONENTS = ("frozen_champion", "simple_rule")
PERIODIC_REASONS = {"periodic", "periodic_collection", "periodic_learning"}


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Malformed metrics JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Metrics line {line_number} is not an object")
            rows.append(row)
    return rows


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _number(value: Any, *, integer: bool = False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0
            and (not integer or int(value) == value))


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _shapes(row: dict[str, Any]) -> set[str]:
    shapes = {"inference:" + str(key)
              for key in row.get("padding_cycle", {}).get("shapes", {})}
    for key in row.get("learner_shapes", {}):
        shapes.add("learner:" + str(key))
    if not row.get("learner_shapes"):
        shapes.update("learner_history:" + str(key) for key in row.get("history_buckets", {}))
    return shapes


def summarize_rows(
    rows: list[dict[str, Any]], *, config: dict[str, Any] | None = None,
    warmup_cycles: int = 25, process_returncode: int | None = 0,
    process_wall_seconds: float | None = None,
    require_evaluation: bool = True, require_periodic_checkpoint: bool = True,
) -> dict[str, Any]:
    """Return JSON-safe evidence; ``eligible`` means usable in this comparison.

    Warmup counts cycles in the last invocation, never absolute checkpoint
    cycle numbers. Only a normal, successful terminal save permits trimming a
    time-limited trailing partial cycle. Missing or partial interior cycles,
    nonfinite telemetry and nonzero process exit codes invalidate a trial.
    Old imported checkpoint logs do not contribute to rates or warmup.
    """
    if warmup_cycles < 0:
        raise ValueError("warmup_cycles must be nonnegative")
    train_config = (config or {}).get("train", {})
    expected_updates = train_config.get("updates_per_cycle", 4)
    effective_batch = train_config.get("micro_batch", 32) * train_config.get("accumulation", 8)
    eval_every = train_config.get("eval_every", 25)
    eval_deals = train_config.get("eval_deals", 32)
    reasons: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {
        "schema": 1, "eligible": False, "reasons": reasons, "warnings": warnings,
        "process_returncode": process_returncode,
        "primary_metric": "successful_updates_per_full_cycle_second",
        "interpretation": "Measured workload throughput; not playing strength or a global hardware optimum.",
        "window": {}, "totals": {}, "rates": {}, "padding": {}, "excluded": {},
    }
    if process_returncode != 0:
        reasons.append("training_process_did_not_exit_successfully")
    starts = [i for i, row in enumerate(rows) if row.get("event") == "start"]
    initial_saves = [i for i, row in enumerate(rows)
                     if row.get("event") == "checkpoint" and row.get("reason") == "session_start"]
    if not starts or (initial_saves and initial_saves[-1] > starts[-1]):
        reasons.append("latest_session_has_no_training_start")
        return result
    start_index = starts[-1]
    start = rows[start_index]
    session_id = start.get("session_id")
    session = rows[start_index:]
    if session_id is not None:
        if any(row.get("session_id") not in (None, session_id) for row in session):
            reasons.append("mixed_session_ids_after_latest_start")
        # Legacy rows can lack an ID; the explicit start boundary is still authoritative.
    if any(not _finite(row) for row in session):
        reasons.append("nonfinite_metrics_in_latest_session")
        return result  # Never put NaN/Infinity into the returned JSON.
    if any(row.get("nonfinite_steps", 0) > 0 for row in session):
        reasons.append("nonfinite_optimizer_steps_in_latest_session")
    terminal_saves = [row for row in session
                      if row.get("event") == "checkpoint" and row.get("reason") == "session_end"]
    normal_end = bool(terminal_saves) and process_returncode == 0
    if not terminal_saves:
        reasons.append("no_successful_session_end_checkpoint_record")
    if any(row.get("event") == "checkpoint" and row.get("reason") == "error" for row in session):
        reasons.append("session_ended_with_error_checkpoint")
        normal_end = False
    if terminal_saves and terminal_saves[-1].get("remote_ok") is False:
        reasons.append("terminal_remote_checkpoint_failed")
        normal_end = False
    if terminal_saves:
        result["terminal_checkpoint_counters"] = {
            key: terminal_saves[-1].get(key) for key in ("cycle", "updates", "frames")}

    records = []
    pending = []
    previous_cycle = start.get("cycle")
    previous_updates = start.get("updates")
    previous_frames = start.get("frames")
    for row in session[1:]:
        if row.get("event") in ("session_end", "checkpoint") and row.get("reason") == "session_end":
            continue
        pending.append(row)
        if row.get("event") != "cycle_end":
            continue
        cycle = row.get("cycle")
        matching = [item for item in pending
                    if item.get("event") == "train" and item.get("cycle") == cycle]
        record = {"end": row, "events": pending, "train": matching[0] if len(matching) == 1 else {},
                  "issues": [], "ordinal": len(records) + 1}
        issues = record["issues"]
        if not _number(cycle, integer=True) or (
                _number(previous_cycle, integer=True) and cycle != previous_cycle + 1):
            issues.append("nonconsecutive_cycle_counter")
        if len(matching) != 1 or len([item for item in pending if item.get("event") == "train"]) != 1:
            issues.append("missing_or_duplicate_matching_train_row")
        train = record["train"]
        updates = row.get("updates_completed")
        if not _number(updates, integer=True) or updates != train.get("successful_steps"):
            issues.append("successful_update_counter_mismatch")
        elif updates != expected_updates:
            issues.append("incomplete_learning_cycle")
        for name in ("full_cycle_seconds", "collection_seconds", "learning_seconds",
                     "evaluation_seconds", "checkpoint_seconds", "selfplay_frames"):
            if not _number(row.get(name), integer=name == "selfplay_frames"):
                issues.append("invalid_" + name)
        if _number(row.get("full_cycle_seconds")):
            if row["full_cycle_seconds"] <= 0:
                issues.append("empty_cycle_duration")
            for name in ("collection_seconds", "learning_seconds", "evaluation_seconds", "checkpoint_seconds"):
                if _number(row.get(name)) and row[name] > row["full_cycle_seconds"] + 0.01:
                    issues.append("phase_exceeds_full_cycle_duration:" + name)
        if not _number(train.get("learner_seconds")) or (
                _number(updates) and updates > 0 and train.get("learner_seconds", 0) <= 0):
            issues.append("invalid_learner_seconds")
        for counter, previous, delta in (("updates", previous_updates, updates),
                                         ("frames", previous_frames, row.get("selfplay_frames"))):
            if _number(previous, integer=True) and _number(delta, integer=True):
                if row.get(counter) != previous + delta or train.get(counter) != row.get(counter):
                    issues.append("global_" + counter + "_counter_mismatch")
        if not _number(train.get("fresh_samples"), integer=True):
            issues.append("invalid_fresh_samples")
        due = bool(eval_every > 0 and _number(cycle, integer=True) and cycle % eval_every == 0)
        evaluations = [item for item in pending if item.get("event") == "evaluation"]
        if due:
            for opponent in EXPECTED_OPPONENTS:
                completed = [item for item in evaluations if item.get("opponent") == opponent
                             and item.get("result", {}).get("deals") == eval_deals
                             and item.get("result", {}).get("games") == 2 * eval_deals]
                if len(completed) != 1:
                    issues.append("incomplete_scheduled_evaluation:" + opponent)
        record["evaluation_due"] = due
        records.append(record)
        pending = []
        previous_cycle, previous_updates, previous_frames = (
            row.get("cycle"), row.get("updates"), row.get("frames"))

    # A time cap can interrupt one last cycle. Never silently drop failed or
    # slow complete cycles, and never use the old imported session as fallback.
    trailing = []
    while records and records[-1]["issues"]:
        record = records[-1]
        trimmable = all(issue == "incomplete_learning_cycle"
                        or issue.startswith("incomplete_scheduled_evaluation:")
                        for issue in record["issues"])
        stop_reason = record["end"].get("stop_reason")
        if not (normal_end and trimmable and stop_reason == "time_limit"):
            break
        trailing.insert(0, records.pop())
    if trailing:
        warnings.append("time_limited_partial_tail_excluded; inspect excluded time before comparing budgets")
    if any(row.get("event") == "train" for row in pending):
        reasons.append("unterminated_train_row_without_cycle_end")
    for record in records:
        if record["issues"]:
            reasons.append("cycle_" + str(record["end"].get("cycle")) + ":" + ",".join(record["issues"]))
    warmup = records[:warmup_cycles]
    measured = records[warmup_cycles:]
    result["window"] = {
        "session_id": session_id, "session_start_cycle": start.get("cycle"),
        "source_rows_ignored": start_index, "warmup_cycles_requested": warmup_cycles,
        "warmup_cycles_observed": len(warmup), "measured_cycles": len(measured),
        "first_cycle": measured[0]["end"].get("cycle") if measured else None,
        "last_cycle": measured[-1]["end"].get("cycle") if measured else None,
        "incomplete_terminal_cycles_excluded": len(trailing),
    }
    initial_save = rows[initial_saves[-1]] if initial_saves else {}
    result["excluded"] = {
        "session_start_checkpoint_seconds": initial_save.get("checkpoint_seconds", 0),
        "session_end_checkpoint_seconds": sum(row.get("checkpoint_seconds", 0) for row in terminal_saves),
        "warmup_full_cycle_seconds": sum(r["end"].get("full_cycle_seconds", 0) for r in warmup
                                          if _number(r["end"].get("full_cycle_seconds"))),
        "incomplete_tail_full_cycle_seconds": sum(r["end"]["full_cycle_seconds"] for r in trailing),
        "process_wall_seconds": process_wall_seconds,
    }
    if _number(process_wall_seconds):
        timed = sum(r["end"].get("full_cycle_seconds", 0) for r in records + trailing
                    if _number(r["end"].get("full_cycle_seconds")))
        result["excluded"]["process_time_outside_logged_cycles_seconds"] = max(0., process_wall_seconds - timed)
    session_ends = [row for row in session if row.get("event") == "session_end"]
    if session_ends and _number(session_ends[-1].get("runtime_seconds")):
        result["session_runtime_seconds"] = session_ends[-1]["runtime_seconds"]
    if not measured:
        reasons.append("no_complete_cycles_after_session_local_warmup")
        return result
    if any(record["issues"] for record in measured):
        return result

    total = result["totals"]
    for name in ("full_cycle_seconds", "collection_seconds", "learning_seconds",
                 "evaluation_seconds", "checkpoint_seconds", "selfplay_frames"):
        total[name] = sum(record["end"][name] for record in measured)
    total["successful_updates"] = sum(record["end"]["updates_completed"] for record in measured)
    total["learner_seconds"] = sum(record["train"]["learner_seconds"] for record in measured)
    total["fresh_complete_samples"] = sum(record["train"]["fresh_samples"] for record in measured)
    total["learning_rows"] = total["successful_updates"] * effective_batch
    total["effective_batch"] = effective_batch
    total["completed_evaluation_pairs"] = sum(record["evaluation_due"] for record in measured)
    total["periodic_checkpoints"] = sum(
        event.get("event") == "checkpoint" and event.get("reason") in PERIODIC_REASONS
        for record in measured for event in record["events"])
    if total["successful_updates"] <= 0:
        reasons.append("no_successful_updates_in_measurement_window")
    if total["selfplay_frames"] <= 0 or total["fresh_complete_samples"] <= 0:
        reasons.append("no_representative_collection; credit_only_window_is_insufficient")
    if require_evaluation and (eval_every <= 0 or not total["completed_evaluation_pairs"]):
        reasons.append("no_complete_scheduled_evaluation_pair_in_measurement_window")
    if require_periodic_checkpoint and (not total["periodic_checkpoints"] or total["checkpoint_seconds"] <= 0):
        reasons.append("no_periodic_checkpoint_inside_measurement_window")
    result["rates"] = {
        "successful_updates_per_full_cycle_second": _ratio(total["successful_updates"], total["full_cycle_seconds"]),
        "learning_rows_per_full_cycle_second": _ratio(total["learning_rows"], total["full_cycle_seconds"]),
        "selfplay_decisions_per_full_cycle_second": _ratio(total["selfplay_frames"], total["full_cycle_seconds"]),
        "selfplay_decisions_per_collection_second": _ratio(total["selfplay_frames"], total["collection_seconds"]),
        "learner_only_updates_per_second": _ratio(total["successful_updates"], total["learner_seconds"]),
        "learner_only_rows_per_second": _ratio(total["learning_rows"], total["learner_seconds"]),
        "learning_rows_per_fresh_complete_sample": _ratio(total["learning_rows"], total["fresh_complete_samples"]),
    }
    padding = {name: 0 for name in PADDING_COUNTS}
    padding["shapes"] = {}
    for record in measured:
        raw = record["train"].get("padding_cycle")
        if not isinstance(raw, dict):
            warnings.append("missing_aggregate_padding_cycle; last-call padding is not substituted")
            continue
        for name in PADDING_COUNTS:
            if not _number(raw.get(name, 0), integer=True):
                reasons.append("invalid_padding_count:" + name)
            else:
                padding[name] += raw.get(name, 0)
        for shape, count in raw.get("shapes", {}).items():
            if _number(count, integer=True):
                padding["shapes"][shape] = padding["shapes"].get(shape, 0) + count
            else:
                reasons.append("invalid_padding_shape_count")
    for label, used, slots in (("action", "legal_actions", "action_slots"),
                               ("history", "history_tokens", "history_slots"),
                               ("state", "requests", "state_slots")):
        if padding[used] > padding[slots]:
            reasons.append("padding_used_exceeds_slots:" + label)
        padding[label + "_padding_fraction"] = (1 - padding[used] / padding[slots]) if padding[slots] else None
    result["padding"] = padding
    warm_shapes = set().union(*(_shapes(record["train"]) for record in warmup)) if warmup else set()
    measured_shapes = set().union(*(_shapes(record["train"]) for record in measured))
    result["observed_static_shapes"] = sorted(measured_shapes)
    result["static_shapes_first_seen_after_warmup"] = sorted(measured_shapes - warm_shapes)
    if measured_shapes - warm_shapes:
        warnings.append("new_static_shapes_after_warmup; their compile cost remains included")
    result["warnings"] = list(dict.fromkeys(warnings))
    result["reasons"] = list(dict.fromkeys(reasons))
    result["eligible"] = not reasons
    return result


def summarize_metrics(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    return summarize_rows(read_metrics(path), **kwargs)


def rank_trials(trials: list[dict[str, Any]], *, required_repeats: int = 3) -> dict[str, Any]:
    """Pair each repeat with its own baseline; never average per-cycle rates."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        groups.setdefault(trial["candidate"], []).append(trial)
    baseline = {trial["repeat"]: trial for trial in groups.get("baseline", [])
                if trial.get("summary", {}).get("eligible") and trial.get("returncode") == 0}
    ranking = []
    for candidate, records in groups.items():
        counts = {}
        for trial in records:
            counts[trial["repeat"]] = counts.get(trial["repeat"], 0) + 1
        eligible = [trial for trial in records if trial.get("returncode") == 0
                    and trial.get("summary", {}).get("eligible") and not trial.get("diagnostic_only")
                    and counts[trial["repeat"]] == 1]
        pairs = []
        for trial in eligible:
            reference = baseline.get(trial["repeat"])
            if reference is None:
                continue
            rate = trial["summary"]["rates"]["successful_updates_per_full_cycle_second"]
            reference_rate = reference["summary"]["rates"]["successful_updates_per_full_cycle_second"]
            if rate and reference_rate:
                pairs.append({"repeat": trial["repeat"], "speedup": rate / reference_rate})
        if not pairs:
            continue
        updates = sum(trial["summary"]["totals"]["successful_updates"] for trial in eligible)
        seconds = sum(trial["summary"]["totals"]["full_cycle_seconds"] for trial in eligible)
        speedups = [pair["speedup"] for pair in pairs]
        ranking.append({
            "candidate": candidate, "eligible_repeats": len(eligible), "paired_repeats": len(pairs),
            "pooled_successful_updates_per_full_cycle_second": _ratio(updates, seconds),
            "paired_geometric_mean_speedup": statistics.geometric_mean(speedups),
            "paired_median_speedup": statistics.median(speedups),
            "paired_min_speedup": min(speedups), "paired_max_speedup": max(speedups),
            "pairs": pairs,
        })
    ranking.sort(key=lambda item: (-item["paired_geometric_mean_speedup"], item["candidate"]))
    competitive = {name for name, records in groups.items() if not all(record.get("diagnostic_only") for record in records)}
    complete = (required_repeats >= 3 and bool(ranking)
                and {item["candidate"] for item in ranking} == competitive
                and all(item["paired_repeats"] >= required_repeats for item in ranking))
    return {
        "status": "measured_stage_comparison" if complete else "provisional",
        "ranking": ranking, "required_repeats": max(3, required_repeats),
        "best_observed_candidate": ranking[0]["candidate"] if ranking else None,
        "interpretation": "Paired by repeat and common initial checkpoint; actor trajectories are not identical. "
                          "This is a comparison of tested configurations, not a global optimum or a strength result.",
        "excluded_trials": [{"candidate": trial["candidate"], "repeat": trial["repeat"],
                             "reasons": trial.get("summary", {}).get("reasons", [])}
                            for trial in trials if trial.get("returncode") != 0
                            or not trial.get("summary", {}).get("eligible")
                            or trial.get("diagnostic_only")],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warmup-cycles", type=int, default=25)
    parser.add_argument("--returncode", type=int, required=True,
                        help="Actual train process exit code; final checkpoint failures invalidate a trial.")
    parser.add_argument("--process-wall-seconds", type=float)
    args = parser.parse_args()
    summary = summarize_metrics(args.metrics, config=json.loads(args.config.read_text()),
                                warmup_cycles=args.warmup_cycles, process_returncode=args.returncode,
                                process_wall_seconds=args.process_wall_seconds)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
