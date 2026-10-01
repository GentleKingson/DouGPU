import pickle
import numpy as np
import pytest
from dougpu.config import TrainConfig
from dougpu.encoding import PackedRequests, pack_requests
from dougpu.replay import PackedSamples, Replay, group_history


def rows(lengths):
    return [(np.full(n, (i % 22)+1, np.uint8), np.full(21, i/7),
             np.full(16, i/9), i % 3, np.full(30, i/11), float(i % 2), i)
            for i, n in enumerate(lengths)]


def padded(batch):
    if not len(batch):
        return batch
    tokens = np.zeros((len(batch), 512), np.uint8)
    tokens[:, :batch.data['tokens'].shape[1]] = batch.data['tokens']
    return PackedSamples(dict(batch.data, tokens=tokens))


def same(a, b):
    assert (a.pos, a.size) == (b.pos, b.size)
    for key in a.data:
        np.testing.assert_array_equal(a.data[key], b.data[key])


def test_merge_empty_and_single_batch_contracts():
    empty = PackedSamples.pack([])
    sample = PackedSamples.pack(rows([6]))
    assert len(PackedSamples.merge([empty])) == 0
    assert PackedSamples.merge([empty, sample, empty]) is sample
    request = pack_requests([{'tokens': np.ones(6, np.uint8), 'state': np.zeros(21),
                              'role': 0, 'actions': np.zeros((2, 16))}])
    assert PackedRequests.merge([None]) == []
    assert PackedRequests.merge([None, request]).data is request


@pytest.mark.parametrize('capacity', [1, 17, 64])
def test_compact_merge_ring_wrap_and_legacy_full_width(capacity):
    compact, legacy = Replay(capacity), Replay(capacity)
    # Begin with long histories so later shorter writes must clear stale tails.
    for lengths in ([512]*capacity, [1, 6, 64, 65, 128, 129, 256, 257, 486, 512],
                    [6]*139, [], [64, 65], [1]*capacity):
        samples = rows(lengths)
        batches = [PackedSamples.pack(samples[:3]), PackedSamples.pack([]),
                   PackedSamples.pack(samples[3:])]
        batch = PackedSamples.merge(batches)
        assert len(batch) == len(samples)
        if len(batch):
            assert batch.data['tokens'].shape == (len(samples), max(lengths))
        batch = pickle.loads(pickle.dumps(batch))
        compact.add(batch)
        legacy.add(padded(batch))
        same(compact, legacy)
    restored = Replay(capacity)
    restored.restore(compact.export())
    same(compact, restored)
    assert restored.data['tokens'].shape == (capacity, 512)


def test_mixed_old_new_batches_and_sample_rng():
    batches = [PackedSamples.pack(rows([6, 65, 129])),
               padded(PackedSamples.pack(rows([6, 256, 486])))]
    merged = PackedSamples.merge(batches)
    assert merged.data['tokens'].shape == (6, 512)
    a, b = Replay(32), Replay(32)
    a.add(merged)
    for batch in batches:
        b.add(padded(batch))
    same(a, b)
    cfg = TrainConfig(micro_batch=32, accumulation=8, history_groups=4)
    ra, rb = np.random.default_rng(8), np.random.default_rng(8)
    for _ in range(10):
        xa = group_history(a.sample(cfg, ra, 0), 4)
        xb = group_history(b.sample(cfg, rb, 0), 4)
        for ba, bb in zip(xa, xb):
            for key in ba:
                np.testing.assert_array_equal(ba[key], bb[key])
                assert ba[key].dtype == bb[key].dtype
        assert ra.bit_generator.state == rb.bit_generator.state
    merged.data['tokens'][:] = 99
    same(a, b)


@pytest.mark.parametrize('shape', [(1, 0), (1, 513), (2, 6), (6,), (1, 5)])
def test_bad_compact_tokens_rejected_before_replay_write(shape):
    replay = Replay(3)
    replay.add(rows([512, 512, 512]))
    before = {k: v.copy() for k, v in replay.data.items()}
    batch = PackedSamples.pack(rows([6]))
    batch.data['tokens'] = np.zeros(shape, np.uint8)
    with pytest.raises(ValueError):
        replay.add(batch)
    assert (replay.pos, replay.size) == (0, 3)
    for key in before:
        np.testing.assert_array_equal(before[key], replay.data[key])


def test_preserves_full_width_legacy_padding_bytes():
    batch = padded(PackedSamples.pack(rows([6])))
    batch.data['tokens'][:, 6:] = 19
    replay = Replay(1)
    replay.add(batch)
    np.testing.assert_array_equal(replay.data['tokens'], batch.data['tokens'])
