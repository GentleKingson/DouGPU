"""Keep hardware execution changes separate from training algorithm changes."""

FIXED_FIELDS = ('seed', 'engine', 'lr', 'weight_decay', 'grad_clip', 'ntp_weight',
                'belief_weight', 'epsilon_start', 'epsilon_end', 'epsilon_frames',
                'replay_capacity', 'replay_max_age', 'fresh_samples', 'updates_per_cycle')


def check_training_semantics(old_train, config):
    for key in FIXED_FIELDS:
        if old_train[key] != getattr(config, key):
            raise ValueError('Training semantics changed: ' + key)
    if bool(old_train.get('historical_opponents', [])) != bool(config.historical_opponents):
        raise ValueError('Training semantics changed: historical_opponents')
    if config.historical_opponents and old_train.get('historical_fraction', .5) != config.historical_fraction:
        raise ValueError('Training semantics changed: historical_fraction')
    old_batch = old_train['micro_batch'] * old_train['accumulation']
    if old_batch != config.batch_size:
        raise ValueError(f'Effective batch changed: {old_batch} -> {config.batch_size}')


def check_array_state(saved, model_config):
    """Reject incomplete policy exports or malformed optimizer state without requiring JAX."""
    import numpy as np
    from .replay import Replay
    params, opt, champion = saved['params'], saved['optimizer'], saved['champion']
    expected = expected_parameter_shapes(model_config)
    if set(params) != set(expected) or set(champion) != set(expected):
        raise ValueError('Full training parameters including auxiliary heads are required')
    if set(opt) != {'step', 'm', 'v'} or set(opt['m']) != set(params) or set(opt['v']) != set(params):
        raise ValueError('Incomplete Adam optimizer state')
    for group in (params, champion, opt['m'], opt['v']):
        for name, shape in expected.items():
            value = np.asarray(group[name])
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid training array: ' + name)
    step = np.asarray(opt['step'])
    if step.shape or not np.issubdtype(step.dtype, np.integer) or int(step) != saved['meta']['updates']:
        raise ValueError('Adam step and successful-update count disagree')


def expected_parameter_shapes(cfg):
    from .encoding import STATE_DIM, ACTION_DIM, BELIEF_DIM
    d, f, q, h = cfg.width, cfg.ffn, cfg.q_hidden, cfg.width // cfg.heads
    shapes = {'embedding': (cfg.vocab, d), 'final_norm': (d,), 'ntp': (d, cfg.vocab)}
    for i in range(cfg.layers):
        stem = f'block{i}.'
        for name in ('anorm', 'fnorm'):
            shapes[stem + name] = (d,)
        for name in ('qnorm', 'knorm'):
            shapes[stem + name] = (h,)
        for name in ('q', 'k', 'v', 'o'):
            shapes[stem + name] = (d, d)
        shapes.update({stem + 'gate': (d, f), stem + 'up': (d, f), stem + 'down': (f, d)})
    for name, ni, no in (('state1', d + STATE_DIM, d), ('state2', d, d),
                         ('q1', d + ACTION_DIM, q), ('q2', q, d), ('qout', d, 3),
                         ('belief1', d, d), ('belief2', d, BELIEF_DIM)):
        shapes[name + '.w'], shapes[name + '.b'] = (ni, no), (no,)
    return shapes
