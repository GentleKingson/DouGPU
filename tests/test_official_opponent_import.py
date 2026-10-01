"""Validate the real vendored model import route without installing Torch.

The stub covers class loading and constructor routing only. It cannot validate
official state_dict compatibility, tensor computation, or playing strength.
"""
import importlib.abc
from pathlib import Path
import sys
from types import ModuleType

import pytest

from dougpu.checkpoint import sha256_file
from dougpu.encoding import POSITIONS
from dougpu.evaluate import DouZeroOpponent, _official_model_dict


@pytest.fixture
def torch_stub(monkeypatch):
    torch, nn = ModuleType('torch'), ModuleType('torch.nn')
    calls = {'threads': [], 'loads': [], 'trainer_imports': []}

    class Module:
        def load_state_dict(self, state, strict=True):
            self.stub_state, self.stub_strict = state, strict

        def eval(self):
            self.stub_eval = True
            return self

    nn.Module = Module
    nn.LSTM = nn.Linear = lambda *args, **kwargs: (args, kwargs)
    torch.nn = nn
    torch.set_num_threads = calls['threads'].append

    def load(path, **kwargs):
        calls['loads'].append((Path(path), kwargs))
        return {'stub_weight': .25}

    torch.load = load
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.nn', nn)
    # A stale module cache must not make the old broken package import pass.
    for name in tuple(sys.modules):
        if name == 'douzero.dmc' or name.startswith('douzero.dmc.'):
            monkeypatch.delitem(sys.modules, name)

    class RejectTrainerPackage(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'douzero.dmc' or fullname.startswith('douzero.dmc.'):
                calls['trainer_imports'].append(fullname)
                raise AssertionError('The incomplete upstream trainer package must not be imported')
            return None

    monkeypatch.setattr(sys, 'meta_path', [RejectTrainerPackage(), *sys.meta_path])
    return nn, calls


def test_real_model_file_resolves_all_roles_without_parent_package(torch_stub):
    nn, calls = torch_stub
    root = Path(__file__).resolve().parents[1]/'vendor'/'douzero'/'dmc'
    paths = (root/'models.py', root/'__init__.py')
    before = {str(path): sha256_file(path) for path in paths}
    models = _official_model_dict()
    assert set(models) == set(POSITIONS)
    assert models['landlord'].__name__ == 'LandlordLstmModel'
    assert models['landlord_up'] is models['landlord_down']
    assert models['landlord_up'].__name__ == 'FarmerLstmModel'
    assert all(issubclass(model, nn.Module) for model in models.values())
    assert all(model.__module__ == '_dougpu_official_douzero_models' for model in models.values())
    assert calls['trainer_imports'] == []
    assert 'douzero.dmc' not in sys.modules
    assert before == {str(path): sha256_file(path) for path in paths}


def test_opponent_constructor_uses_standalone_loader_and_safe_cpu_weight_flags(tmp_path, torch_stub):
    _, calls = torch_stub
    for role in POSITIONS:
        (tmp_path/(role+'.ckpt')).write_bytes(b'placeholder read only by the torch stub')
    opponent = DouZeroOpponent(tmp_path)
    assert set(opponent.models) == set(opponent.hashes) == set(POSITIONS)
    assert calls['threads'] == [1]
    assert len(calls['loads']) == 3
    assert {path.stem for path, _ in calls['loads']} == set(POSITIONS)
    assert all(options == {'map_location': 'cpu', 'weights_only': True}
               for _, options in calls['loads'])
    assert all(model.stub_eval and model.stub_strict for model in opponent.models.values())
    assert all(model.stub_state == {'stub_weight': .25} for model in opponent.models.values())
    assert calls['trainer_imports'] == []
    assert 'douzero.dmc' not in sys.modules
