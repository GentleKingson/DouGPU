"""Evidence tests: no accelerator is imported or required."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from dougpu.metrics_summary import rank_trials, summarize_rows
from tune_local import STAGES, make_candidates, run_trial, snapshot_source, verify_latest


CONFIG = {
    "model": {"width": 128, "remat": True},
    "train": {
        "backend": "cuda", "require_tpu": False, "engine": "douzero", "seed": 42,
        "workers": 8, "envs_per_worker": 8, "actor_groups": 2, "infer_batch": 32,
        "eval_workers": 8, "micro_batch": 32, "accumulation": 8, "history_groups": 1,
        "fresh_samples": 2048, "updates_per_cycle": 4, "eval_every": 2, "eval_deals": 32,
        "checkpoint_seconds": 300, "replay_capacity": 65536, "replay_max_age": 64,
        "save_replay": True, "resume": True, "lr": 0.0001, "ntp_weight": 0.02,
        "belief_weight": 0.05, "infer_history_buckets": True, "infer_fused": True,
        "action_chunk": 2048, "learner_remat": True, "attention_impl": "manual",
    },
}


def session(start_cycle=100, cycles=6, session_id="new", durations=None, frames=None):
    rows = []
    cycle, updates, global_frames = start_cycle, start_cycle * 4, 10000

    def row(event, **extra):
        result = {"event": event, "cycle": cycle, "updates": updates, "frames": global_frames, **extra}
        if session_id is not None:
            result["session_id"] = session_id
        rows.append(result)

    row("checkpoint", reason="session_start", checkpoint_seconds=9, remote_ok=None)
    row("start", effective_batch=256)
    for index in range(cycles):
        # Save inside collection logs the PRE-increment cycle; the summary must
        # assign events by interval, not by checkpoint row cycle equality.
        row("checkpoint", reason="periodic_collection", checkpoint_seconds=2, remote_ok=None)
        cycle += 1
        updates += 4
        decisions = (frames or [100] * cycles)[index]
        global_frames += decisions
        duration = (durations or [20.] * cycles)[index]
        row("train", successful_steps=4, fresh_samples=2048 if decisions else 0,
            learner_seconds=4., learning_seconds=5., loss=1., q_loss=0.8, ntp_loss=0.1,
            belief_loss=0.1, grad_norm=0.5, history_buckets={"512": 4},
            learner_shapes={"8x32x512": 4},
            # Deliberately dishonest last-call fractions must not be averaged.
            padding={"action_padding_fraction": 0.99},
            padding_cycle={"requests": decisions, "state_slots": decisions * 2,
                           "legal_actions": decisions * 3, "action_slots": decisions * 4,
                           "history_tokens": decisions * (10 if index % 2 == 0 else 100),
                           "history_slots": decisions * (20 if index % 2 == 0 else 100),
                           "encoder_batches": 1 if decisions else 0,
                           "fused_batches": 1 if decisions else 0, "shapes": {"32x512": 1}})
        if cycle % 2 == 0:
            for opponent in ("frozen_champion", "simple_rule"):
                row("evaluation", opponent=opponent, result={"deals": 32, "games": 64}, evaluation_seconds=1.)
        row("cycle_end", full_cycle_seconds=duration, updates_completed=4,
            selfplay_frames=decisions, collection_seconds=8., learning_seconds=5.,
            evaluation_seconds=2. if cycle % 2 == 0 else 0., checkpoint_seconds=2.,
            stop_reason=None)
    row("checkpoint", reason="session_end", checkpoint_seconds=13., remote_ok=None)
    row("session_end", runtime_seconds=sum(durations or [20.] * cycles) + 22.)
    return rows


def summarize(rows, **kwargs):
    return summarize_rows(rows, config=CONFIG, warmup_cycles=2, **kwargs)


def test_warmup_counts_latest_session_cycles_not_global_checkpoint_counter():
    old = session(start_cycle=0, cycles=40, session_id="old")
    old[3]["loss"] = float("nan")  # Old imported metrics cannot poison the new trial.
    result = summarize(old + session(), process_wall_seconds=150)
    assert result["eligible"], result["reasons"]
    assert result["window"]["session_id"] == "new"
    assert result["window"]["first_cycle"] == 103
    assert result["window"]["measured_cycles"] == 4
    assert result["totals"]["successful_updates"] == 16
    assert result["totals"]["full_cycle_seconds"] == 80
    assert result["excluded"]["warmup_full_cycle_seconds"] == 40
    assert result["excluded"]["session_start_checkpoint_seconds"] == 9
    assert result["excluded"]["session_end_checkpoint_seconds"] == 13
    assert result["excluded"]["process_time_outside_logged_cycles_seconds"] == 30


def test_old_log_format_without_session_ids_is_supported():
    result = summarize(session(start_cycle=0, session_id=None) + session(session_id=None))
    assert result["eligible"]
    assert result["window"]["first_cycle"] == 103


def test_new_session_failing_before_start_does_not_reuse_old_training_rows():
    rows = session() + [{"event": "checkpoint", "reason": "session_start", "cycle": 106}]
    result = summarize(rows, process_returncode=1)
    assert not result["eligible"]
    assert "latest_session_has_no_training_start" in result["reasons"]
    assert not result["rates"]


def test_credit_only_cycle_and_padding_use_aggregate_numerators_denominators():
    rows = session(durations=[20, 20, 10, 30, 10, 50], frames=[100, 100, 100, 0, 200, 100])
    result = summarize(rows)
    assert result["eligible"], result["reasons"]
    assert result["totals"]["selfplay_frames"] == 400
    assert result["totals"]["full_cycle_seconds"] == 100
    assert result["rates"]["successful_updates_per_full_cycle_second"] == 16 / 100
    assert result["rates"]["learner_only_updates_per_second"] == 1
    assert result["rates"]["learning_rows_per_full_cycle_second"] == 4096 / 100
    assert result["rates"]["selfplay_decisions_per_collection_second"] == 400 / 32
    assert result["rates"]["learning_rows_per_fresh_complete_sample"] == 4096 / 6144
    assert result["padding"]["action_padding_fraction"] == 0.25
    assert result["padding"]["history_padding_fraction"] == pytest.approx(1 - 13000 / 16000)


def test_window_of_only_credit_is_insufficient_even_with_updates():
    result = summarize(session(frames=[100, 100, 0, 0, 0, 0]))
    assert not result["eligible"]
    assert any("credit_only_window" in reason for reason in result["reasons"])


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_current_session_rejected_without_nonfinite_summary_output(nonfinite):
    rows = session()
    next(row for row in rows if row["event"] == "train")["loss"] = nonfinite
    result = summarize(rows)
    assert not result["eligible"]
    assert "nonfinite_metrics_in_latest_session" in result["reasons"]
    json.dumps(result, allow_nan=False)


def test_nonfinite_step_in_trimmed_terminal_tail_still_rejects_trial():
    rows = [row for row in session() if not (row["event"] == "evaluation" and row["cycle"] == 106)]
    tail = next(row for row in rows if row["event"] == "cycle_end" and row["cycle"] == 106)
    tail.update(stop_reason="time_limit", nonfinite_steps=1)
    result = summarize(rows)
    assert not result["eligible"]
    assert "nonfinite_optimizer_steps_in_latest_session" in result["reasons"]


@pytest.mark.parametrize("damage", ["missing_train", "global_updates", "missing_rule_eval"])
def test_interior_incomplete_or_unmatched_cycles_rejected(damage):
    rows = session()
    if damage == "missing_train":
        rows = [row for row in rows if not (row["event"] == "train" and row["cycle"] == 103)]
    elif damage == "global_updates":
        next(row for row in rows if row["event"] == "cycle_end" and row["cycle"] == 103)["updates"] += 4
    else:
        rows = [row for row in rows if not (row["event"] == "evaluation" and row["cycle"] == 104
                                          and row["opponent"] == "simple_rule")]
    result = summarize(rows)
    assert not result["eligible"]


def test_time_limit_may_trim_incomplete_terminal_evaluation_but_error_may_not():
    rows = [row for row in session() if not (row["event"] == "evaluation" and row["cycle"] == 106)]
    next(row for row in rows if row["event"] == "cycle_end" and row["cycle"] == 106)["stop_reason"] = "time_limit"
    result = summarize(rows)
    assert result["eligible"], result["reasons"]
    assert result["window"]["last_cycle"] == 105
    assert result["window"]["incomplete_terminal_cycles_excluded"] == 1
    assert result["excluded"]["incomplete_tail_full_cycle_seconds"] == 20
    failed = summarize(rows, process_returncode=1)
    assert not failed["eligible"]
    assert failed["window"]["incomplete_terminal_cycles_excluded"] == 0


def test_session_start_and_terminal_saves_do_not_count_as_periodic_work():
    rows = [row for row in session() if row.get("reason") != "periodic_collection"]
    for row in rows:
        if row["event"] == "cycle_end":
            row["checkpoint_seconds"] = 0
    result = summarize(rows)
    assert not result["eligible"]
    assert "no_periodic_checkpoint_inside_measurement_window" in result["reasons"]


def test_final_save_failure_rejects_otherwise_fast_candidate():
    result = summarize(session(), process_returncode=1)
    assert not result["eligible"]
    assert "training_process_did_not_exit_successfully" in result["reasons"]
    rows = [row for row in session() if row.get("reason") != "session_end"]
    result = summarize(rows)
    assert not result["eligible"]
    assert "no_successful_session_end_checkpoint_record" in result["reasons"]


def test_paired_rank_uses_rates_from_aggregate_time_and_three_repeats():
    trials = []
    for repeat in range(1, 4):
        for candidate, duration in (("baseline", 20.), ("faster", 16.)):
            trials.append({"candidate": candidate, "repeat": repeat, "returncode": 0,
                           "summary": summarize(session(durations=[duration] * 6))})
    one = rank_trials(trials[:2])
    assert one["status"] == "provisional"
    complete = rank_trials(trials)
    assert complete["status"] == "measured_stage_comparison"
    assert complete["best_observed_candidate"] == "faster"
    assert complete["ranking"][0]["paired_geometric_mean_speedup"] == pytest.approx(1.25)
    trials[0]["returncode"] = 1
    incomplete = rank_trials(trials)
    assert incomplete["status"] == "provisional"


def test_diagnostic_attention_is_recorded_but_never_selected():
    base = {"candidate": "baseline", "repeat": 1, "returncode": 0, "summary": summarize(session())}
    diagnostic = copy.deepcopy(base)
    diagnostic.update(candidate="attention_xla", diagnostic_only=True)
    diagnostic["summary"]["rates"]["successful_updates_per_full_cycle_second"] *= 100
    result = rank_trials([base, diagnostic])
    assert result["best_observed_candidate"] == "baseline"
    assert len(result["excluded_trials"]) == 1


def test_duplicate_repeat_does_not_count_as_three_independent_repeats():
    trial = {"candidate": "baseline", "repeat": 1, "returncode": 0, "summary": summarize(session())}
    result = rank_trials([copy.deepcopy(trial) for _ in range(3)])
    assert result["status"] == "provisional"
    assert not result["ranking"]


def test_tuning_stages_preserve_learning_semantics_and_have_valid_geometry():
    invariant_keys = ("seed", "lr", "ntp_weight", "belief_weight", "fresh_samples", "updates_per_cycle",
                      "replay_capacity", "replay_max_age", "eval_every", "eval_deals", "checkpoint_seconds")
    for stage in STAGES:
        for candidate in make_candidates(CONFIG, stage):
            current = candidate["config"]
            assert current["model"] == CONFIG["model"]
            assert current["train"]["micro_batch"] * current["train"]["accumulation"] == 256
            assert current["train"]["accumulation"] % current["train"]["history_groups"] == 0
            for key in invariant_keys:
                assert current["train"][key] == CONFIG["train"][key]
    workers = make_candidates(CONFIG, "workers")
    worker_configs = [item["config"]["train"] for item in workers]
    assert {row["workers"] for row in worker_configs} == {6, 8, 10, 12}
    assert {row["workers"]: row["infer_batch"] for row in worker_configs} == {6: 32, 8: 32, 10: 64, 12: 64}
    assert {item["config"]["train"]["envs_per_worker"] for item in make_candidates(CONFIG, "envs")} == {4, 8, 16, 32}
    assert {item["config"]["train"]["micro_batch"] for item in make_candidates(CONFIG, "learner")} == {16, 32, 64, 128}


def test_failed_preflight_stops_before_training_and_records_oom_without_fallback(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, log_path, _environment):
        commands.append(command[2])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if command[2] == "preflight":
            log_path.write_text("RESOURCE_EXHAUSTED: out of memory\n")
            return 1, 4.
        log_path.write_text("ready\n")
        return 0, 1.

    monkeypatch.setattr("tune_local._run", fake_run)
    args = SimpleNamespace(output=tmp_path / "out", source_run=tmp_path / "source", minutes=10,
                           cycles=1000000, warmup_cycles=2, stage="attention", repeats=3)
    candidate = make_candidates(CONFIG, "attention")[1]
    result = run_trial(args, candidate, 1, {"sha256": "fixed_source_digest"}, {})
    assert commands == ["prepare", "import-checkpoint", "preflight"]
    assert result["failed_step"] == "preflight"
    assert result["returncode"] == 1
    assert not result["summary"]["eligible"]
    assert result["failure_class"] == "out_of_memory_reported; no_configuration_fallback"


def write_checkpoint(root: Path, cycle: int):
    root.mkdir(parents=True, exist_ok=True)
    meta = {"schema": 1, "cycle": cycle, "updates": cycle * 4, "source_lock": {"encoding_schema": 1}}
    files = {key: b"synthetic arrays" for key in ("params.npz", "optimizer.npz", "champion.npz", "replay.npz")}
    files["meta.json"] = json.dumps(meta).encode()
    manifest = {key: hashlib.sha256(value).hexdigest() for key, value in files.items()}
    archive = root / f"ckpt_{cycle:09d}.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for key, value in files.items():
            bundle.writestr(key, value)
        bundle.writestr("manifest.json", json.dumps(manifest))
    marker = root / f"ckpt_{cycle:09d}.ok.json"
    marker.write_text(json.dumps({"file": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                                  "created_ns": cycle, "cycle": cycle, "updates": cycle * 4}))
    return archive, marker


def test_snapshot_verifies_manifest_and_copies_one_generation_read_only(tmp_path):
    source = tmp_path / "source"
    first, first_marker = write_checkpoint(source / "checkpoints", 1)
    broken, _ = write_checkpoint(source / "checkpoints", 2)
    broken.write_bytes(b"bad archive")
    before = {path.name: path.read_bytes() for path in (source / "checkpoints").iterdir()}
    evidence = snapshot_source(source, tmp_path / "output" / "snapshot")
    destination = tmp_path / "output" / "snapshot" / "checkpoints"
    assert evidence["cycle"] == 1
    assert (destination / first.name).read_bytes() == first.read_bytes()
    assert (destination / first_marker.name).read_bytes() == first_marker.read_bytes()
    assert len(list(destination.iterdir())) == 2
    assert {path.name: path.read_bytes() for path in (source / "checkpoints").iterdir()} == before
    assert verify_latest(destination)["meta"]["cycle"] == 1
    with pytest.raises(FileExistsError):
        snapshot_source(source, tmp_path / "output" / "snapshot")
