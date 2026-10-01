from dataclasses import replace
import numpy as np
import pytest
from dougpu.encoding import (PublicState, counts, encode, oracle_label, state_features,
                             action_features, _action_counts)


def public(role=0, hand=(3, 3, 4, 20)):
    return PublicState(role, hand, (4, 3, 2), (3, 4, 5), (), ((), (3,), (3, 3)))


@pytest.mark.parametrize('role', [0, 1, 2])
def test_cached_hand_features_labels_and_output_isolation(role):
    hands = [[3, 3, 4, 20], [6, 7, 30], [8, 8]]
    state = public(role, hands[role])
    expected_state = np.concatenate([counts(state.hand), np.asarray(state.remaining, np.float32)/20,
                                     np.eye(3, dtype=np.float32)[role]])
    expected_label = np.concatenate([counts(hands[(role+k) % 3]) for k in (1, 2)])
    _action_counts.cache_clear()
    for _ in range(3):
        features, label = state_features(state), oracle_label(hands, role)
        np.testing.assert_array_equal(features, expected_state)
        np.testing.assert_array_equal(label, expected_label)
        assert features.dtype == label.dtype == np.float32
        features[:] = -9
        label[:] = -9
    assert _action_counts.cache_info().hits >= 6
    assert not _action_counts(tuple(state.hand)).flags.writeable
    actions = action_features(state.legal)
    for i, action in enumerate(state.legal):
        np.testing.assert_array_equal(actions[i], np.r_[counts(action), float(not action)])


@pytest.mark.parametrize('hand', [(3,)*5, (20, 20), (99,)])
def test_cached_hands_keep_invalid_card_checks(hand):
    _action_counts.cache_clear()
    for _ in range(2):
        with pytest.raises(ValueError):
            state_features(public(hand=hand))
        with pytest.raises(ValueError):
            oracle_label([[], list(hand), []], 0)


def test_mutated_hidden_hands_do_not_change_public_inputs_or_poison_cache():
    state = public(hand=[3, 3, 4, 20])
    hands = [state.hand, [6, 7, 30], [8, 8]]
    before = encode(state)
    old_label = oracle_label(hands, 0)
    hands[1][0] = 9
    label = oracle_label(hands, 0)
    assert not np.array_equal(old_label, label)
    np.testing.assert_array_equal(label, np.r_[counts(hands[1]), counts(hands[2])])
    for key, value in before.items():
        np.testing.assert_array_equal(encode(state)[key], value)
    state.hand[0] = 5
    np.testing.assert_array_equal(state_features(state)[:15], counts(state.hand))
    np.testing.assert_array_equal(state_features(replace(state, hand=tuple(reversed(state.hand))))[:15],
                                  counts(state.hand))
    np.testing.assert_array_equal(_action_counts((3, 3, 4, 20)), counts((3, 3, 4, 20)))


def test_hand_cache_reuses_existing_bounded_cache():
    _action_counts.cache_clear()
    action_features([(3, 4)])
    before = _action_counts.cache_info()
    state_features(public(hand=(3, 4)))
    after = _action_counts.cache_info()
    assert after.hits == before.hits+1 and after.misses == before.misses
    assert after.maxsize == 32768
    assert counts((3, 4)).flags.writeable
