"""Evaluate an exported policy against a simple rule or official DouZero weights.

The optional DouZero opponent runs on CPU and is not imported during training.
Provide trusted official state_dict checkpoints; weights are not bundled.
"""
import argparse
import json
from pathlib import Path
from .checkpoint import load_policy, sha256_file
from .config import ModelConfig, TrainConfig
from .runtime import configure_runtime, verify_backend, package_versions


def _official_model_dict():
    """Load the pinned model file without importing the upstream trainer package."""
    import importlib.util
    from bootstrap import PINNED_COMMIT, verify_upstream_files
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root/'upstream_cache'/'source_lock.json').read_text())
    if (lock.get('engine') != 'douzero' or lock.get('encoding_schema') != 1
            or lock.get('douzero_commit') != PINNED_COMMIT):
        raise RuntimeError('Unexpected official-opponent rule source')
    verify_upstream_files(root/'vendor', lock)
    # dmc/__init__.py imports trainer modules intentionally absent from this
    # rules/model-only bundle. Execute the verified, standalone models.py.
    spec = importlib.util.spec_from_file_location(
        '_dougpu_official_douzero_models', root/'vendor'/'douzero'/'dmc'/'models.py')
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load the verified official DouZero model file')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model_dict


class DouZeroOpponent:
    def __init__(self, weights):
        import torch
        model_dict = _official_model_dict()
        from .encoding import POSITIONS
        torch.set_num_threads(1)
        self.models = {}
        self.hashes = {}
        for position in POSITIONS:
            file = Path(weights)/(position+'.ckpt')
            if not file.is_file():
                raise FileNotFoundError(f'Required trusted official weight: {file}')
            net = model_dict[position]()
            # Do not fall back to unsafe arbitrary pickle loading.
            state = torch.load(file, map_location='cpu', weights_only=True)
            net.load_state_dict(state, strict=True)
            if any(not torch.isfinite(value).all().item() for value in net.state_dict().values()):
                raise ValueError(f'Non-finite opponent weight: {file}')
            self.models[position] = net.eval()
            self.hashes[position] = sha256_file(file)

    def choose_games(self, games):
        import torch
        from douzero.env.env import get_obs
        result = []
        with torch.inference_mode():
            for game in games:
                if not hasattr(game, 'game'):
                    raise ValueError('Official baseline evaluation requires the DouZero engine')
                info = game.full_infoset()
                obs = get_obs(info)
                net = self.models[info.player_position]
                output = net(torch.from_numpy(obs['z_batch']).float(),
                             torch.from_numpy(obs['x_batch']).float(), return_value=True)
                index = int(torch.argmax(output['values']).item())
                move = tuple(sorted(obs['legal_actions'][index]))
                result.append(game.legal.index(move))
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--opponent', choices=['rule', 'random', 'douzero'], default='rule')
    ap.add_argument('--weights', default='')
    ap.add_argument('--engine', choices=['douzero', 'reference'], default='douzero')
    ap.add_argument('--deals', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=910001, help='Separate from training selection deals')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--action-chunk', type=int, default=2048)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--backend', choices=['cpu', 'cuda', 'tpu'], default='cuda')
    ap.add_argument('--attention-impl', choices=['manual', 'xla', 'cudnn'], default='manual')
    ap.add_argument('--output', default='evaluation.json')
    args = ap.parse_args()
    if args.deals < 1:
        raise ValueError('--deals must be positive')
    from .evaluation import validate_holdout_seed
    params, meta = load_policy(args.policy)
    selection_seeds = meta.get('metadata', {}).get('selection_seeds', [])
    validate_holdout_seed(args.seed, selection_seeds)
    runtime = TrainConfig(backend='cpu' if args.cpu else args.backend,
                          attention_impl=args.attention_impl, require_tpu=False).validate()
    configure_runtime(runtime)
    import jax
    from .inference import Inference
    from .evaluation import paired_evaluate
    verify_backend(runtime, jax)
    mc = ModelConfig(**meta['model']).validate()
    params = jax.device_put(params)
    infer = Inference(mc, args.batch, args.action_chunk, attention_impl=args.attention_impl)
    opponent = DouZeroOpponent(args.weights) if args.opponent == 'douzero' else args.opponent
    # 1,000 deals creates 2,000 games. This is a deliberate offline evaluation
    # workload, separate from the small in-training monitoring evaluation.
    result = paired_evaluate(params, opponent, infer, args.engine, args.deals, args.seed)
    doc = {'result': result, 'opponent': args.opponent, 'engine': args.engine,
           'policy_sha256': sha256_file(args.policy), 'policy_meta': meta,
           'opponent_hashes': getattr(opponent, 'hashes', {}), 'jax': jax.__version__,
           'packages': package_versions(), 'attention_impl': args.attention_impl,
           'backend': jax.default_backend(), 'note': 'Paired deal bootstrap; fixed-seed evaluation, not an Elo rating'}
    doc['selection_seeds_known'] = bool(selection_seeds)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc, indent=2), flush=True)


if __name__ == '__main__':
    main()
