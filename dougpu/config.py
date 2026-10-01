from dataclasses import dataclass, field

@dataclass(frozen=True)
class ModelConfig:
    width: int = 128
    layers: int = 3
    heads: int = 2
    ffn: int = 512
    q_hidden: int = 256
    max_seq: int = 512
    vocab: int = 32
    bf16: bool = True
    remat: bool = True

    def validate(self):
        if min(self.width, self.layers, self.heads, self.ffn, self.q_hidden) < 1:
            raise ValueError("All model sizes must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width/heads must be a positive even integer")
        if self.max_seq != 512:
            raise ValueError("Version 1 uses a full 512-token history; no silent truncation")
        if self.vocab < 23:
            raise ValueError("Vocabulary is too small")
        return self

@dataclass
class TrainConfig:
    engine: str = "douzero"  # reference is an explicitly opt-in offline test engine
    seed: int = 42
    workers: int = 2
    envs_per_worker: int = 32
    infer_batch: int = 64
    action_chunk: int = 2048
    infer_history_buckets: bool = False  # opt-in; validate on the target TPU first
    infer_fused: bool = False  # single-chunk encoder/scorer/argmax fast path
    eval_optimized_inference: bool = False  # separate experiment from self-play
    learner_remat: bool | None = None  # runtime override; preserve persisted ModelConfig
    actor_groups: int = 1  # ordered groups overlap CPU work with the next inference
    learner_prefetch: bool = False  # at most one future batch; no extra optimizer steps
    selfplay_kv_cache: bool = False
    historical_opponents: list[str] = field(default_factory=list)
    historical_fraction: float = 0.5
    history_groups: int = 1
    ready_first: bool = False
    sample_credit: bool = False
    actor_learner_overlap: bool = False
    replay_version_updates: bool = False
    micro_batch: int = 32
    accumulation: int = 8
    replay_capacity: int = 65536
    replay_max_age: int = 64  # learner cycles, not seconds
    fresh_samples: int = 2048
    updates_per_cycle: int = 4
    lr: float = 0.0001
    weight_decay: float = 0.00001
    grad_clip: float = 1.0
    ntp_weight: float = 0.02
    belief_weight: float = 0.05
    epsilon_start: float = 0.10
    epsilon_end: float = 0.03
    epsilon_frames: int = 1000000
    max_hours: float = 5.5
    max_cycles: int = 100000000
    checkpoint_seconds: float = 300.0
    keep_checkpoints: int = 3
    eval_every: int = 25
    eval_deals: int = 256  # validation only; final strength claims need independent deals
    eval_seed: int = 900001
    promotion_role_margin: float = 0.05  # predeclared absolute win-rate noninferiority margin
    eval_workers: int = 0  # 0 keeps serial evaluation; otherwise borrow idle actors
    eval_bucket_batch: bool = True
    eval_kv_cache: bool = False
    require_tpu: bool = True  # legacy notebook configuration; backend takes precedence
    backend: str | None = None  # explicit cpu/cuda/tpu; never an automatic CPU fallback
    attention_impl: str = "manual"  # execution choice, not persisted ModelConfig
    resume: bool = True
    save_replay: bool = True
    worker_timeout: float = 300.0
    log_every: int = 10

    def validate(self):
        if self.backend not in (None, 'cpu', 'cuda', 'tpu'):
            raise ValueError('backend must be cpu, cuda or tpu')
        if self.attention_impl not in ('manual', 'xla', 'cudnn'):
            raise ValueError('attention_impl must be manual, xla or cudnn')
        if self.attention_impl == 'cudnn' and self.backend != 'cuda':
            raise ValueError('cuDNN attention requires an explicit CUDA backend')
        if self.attention_impl != 'manual' and (self.selfplay_kv_cache or self.eval_kv_cache):
            raise ValueError('Alternate attention is not supported by the research KV path')
        if (not isinstance(self.historical_opponents, list)
                or any(not isinstance(p, str) or not p for p in self.historical_opponents)
                or len(set(self.historical_opponents)) != len(self.historical_opponents)):
            raise ValueError('historical_opponents must be a list of distinct policy paths')
        if not 0 < self.historical_fraction <= 1:
            raise ValueError('historical_fraction must be in (0, 1]')
        if self.historical_opponents and (self.ready_first or self.selfplay_kv_cache):
            raise ValueError('Historical opponents require ordered actors without the research KV cache')
        for k in ("workers", "envs_per_worker", "infer_batch", "action_chunk", "micro_batch",
                  "accumulation", "replay_capacity", "fresh_samples", "updates_per_cycle",
                  "keep_checkpoints", "log_every"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be positive")
        if self.engine not in ("douzero", "reference"):
            raise ValueError("engine must be douzero or reference")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Invalid exploration schedule")
        if self.batch_size < 3 or self.epsilon_frames < 1 or self.replay_max_age < 0:
            raise ValueError("Effective batch must include all 3 roles; check exploration/replay limits")
        if self.max_cycles < 1 or self.lr <= 0 or self.grad_clip <= 0:
            raise ValueError("Invalid optimizer or cycle limit")
        if self.max_hours <= 0 or self.checkpoint_seconds <= 0 or self.eval_deals < 1:
            raise ValueError("Invalid time or evaluation limit")
        if not 0 <= self.eval_workers <= self.workers:
            raise ValueError('eval_workers must be between zero and workers')
        if not 0 <= self.promotion_role_margin < .5:
            raise ValueError('promotion_role_margin must be in [0, .5)')
        if self.actor_groups < 1 or self.workers % self.actor_groups:
            raise ValueError('actor_groups must be a positive divisor of workers')
        if self.worker_timeout <= 0:
            raise ValueError('worker_timeout must be positive')
        if self.history_groups < 1 or self.accumulation % self.history_groups:
            raise ValueError('history_groups must divide accumulation')
        if self.ready_first and not (self.sample_credit and self.replay_version_updates):
            raise ValueError('Ready-first requires sample credits and successful-update versions')
        if self.actor_learner_overlap and not self.ready_first:
            raise ValueError('Actor/learner overlap requires ready-first scheduling')
        if self.learner_remat is not None and not isinstance(self.learner_remat, bool):
            raise ValueError('learner_remat must be null, true or false')
        if (self.infer_history_buckets or self.infer_fused) and self.selfplay_kv_cache:
            raise ValueError('Optimized inference cannot be combined with the research KV cache')
        if self.eval_optimized_inference and self.eval_kv_cache:
            raise ValueError('Optimized evaluation cannot be combined with the research KV cache')
        return self

    @property
    def batch_size(self):
        return self.micro_batch * self.accumulation
