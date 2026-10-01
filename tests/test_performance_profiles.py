import json
from pathlib import Path
from dougpu.config import ModelConfig, TrainConfig
from dougpu.semantics import check_training_semantics


def test_measured_rtx_profile_preserves_learning_semantics():
    root = Path(__file__).resolve().parents[1] / 'configs'
    before = json.loads((root / 'rtx5070_balanced.json').read_text())
    after = json.loads((root / 'rtx5070_throughput.json').read_text())
    assert before['model'] == after['model']
    ModelConfig(**after['model']).validate()
    tc = TrainConfig(**after['train']).validate()
    check_training_semantics(before['train'], tc)
    changed = {k for k, v in before['train'].items() if after['train'][k] != v}
    assert changed == {'accumulation', 'micro_batch', 'history_groups', 'learner_remat',
                       'attention_impl', 'envs_per_worker', 'infer_batch'}
    assert tc.checkpoint_seconds == 300
