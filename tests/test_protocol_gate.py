"""Run without NumPy, JAX or actors: python -m unittest discover -s tests -p test_protocol_gate.py."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.protocol_gate import boundaries, compare, evaluate
from dougpu.config import ModelConfig, TrainConfig


class ProtocolGateTests(unittest.TestCase):
    def setUp(self):
        self.actual = []
        for start, end in ((0, 8), (8, 2000), (2000, 20000)):
            self.actual.append(dict(session_id=str(start), start_updates=start, end_updates=end,
                stop_reason='target_updates', initialization='fresh' if start == 0 else 'resume_previous',
                model=asdict(ModelConfig()), train=asdict(TrainConfig(target_updates=end)),
                worker_seeds='RECORDED', runtime_source='ATTESTED', resume_digest='RECORDED',
                versions={'python': 'fixture'}, source_sha256='fixture', source_lock={'engine': 'fixture'}))
        self.plan = dict(allowed_difference={'ntp_weight': {'A': .02, 'B': 0}}, sessions=deepcopy(self.actual))
        for session in self.plan['sessions']:
            session['train']['ntp_weight'] = 0

    def test_matching_protocol(self):
        self.assertEqual(len(compare(self.actual, self.plan)), 3)
        self.assertEqual(evaluate(self.actual, self.plan)['status'], 'PASS')

    def test_missing_execution_evidence(self):
        for key in ('worker_seeds', 'runtime_source', 'resume_digest'):
            actual = deepcopy(self.actual)
            del actual[1][key]
            self.assertEqual(evaluate(actual, self.plan)['status'], 'FAIL')

    def test_old_missing_eight_update_boundary(self):
        self.plan['sessions'].pop(0)
        self.plan['sessions'][0]['start_updates'] = 0
        with self.assertRaisesRegex(ValueError, 'boundary'):
            compare(self.actual, self.plan)

    def test_missing_or_conflicting_config_and_resume(self):
        for field in ('workers', 'seed', 'log_every', 'max_hours', 'historical_fraction'):
            with self.subTest(field=field):
                plan = deepcopy(self.plan)
                plan['sessions'][0]['train'][field] += 1
                with self.assertRaises(ValueError):
                    compare(self.actual, plan)
        plan = deepcopy(self.plan)
        del plan['sessions'][0]['train']['workers']
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            compare(self.actual, plan)
        self.plan['sessions'][1]['initialization'] = 'fresh'
        with self.assertRaisesRegex(ValueError, 'initialization'):
            compare(self.actual, self.plan)

    def test_boundary_gaps_duplicates_and_interleaving(self):
        rows = [dict(event='train', session_id='a', updates=4, successful_steps=4),
                dict(event='train', session_id='a', updates=8, successful_steps=4),
                dict(event='train', session_id='b', updates=12, successful_steps=4)]
        self.assertEqual([(x['start_updates'], x['end_updates']) for x in boundaries(rows)], [(0, 8), (8, 12)])
        for bad in (rows[:1] + rows, rows + rows[:1], rows[1:], []):
            with self.assertRaises(ValueError):
                boundaries(bad)

    def test_cli_never_accepts_missing_evidence(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/protocol_gate.py'
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / 'plan.json'
            plan.write_text(json.dumps(self.plan))
            for flags in ([], ['-O']):
                result = subprocess.run([sys.executable, *flags, str(script), tmp, str(plan)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(json.loads(result.stdout)['status'], 'FAIL')

    def test_chain_intent_check_survives_optimized_python(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/check_session_chain.py'
        from scripts.protocol_gate import digest
        intent = dict(sessions=[dict(start_updates=0)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = json.dumps(intent).encode()
            (root/'frozen-intent.json').write_bytes(raw)
            for case, message in (('hash', 'hash mismatch'), ('count', 'session count mismatch'),
                                  ('config', 'session mismatch')):
                plan = dict(intent_sha256=digest(raw), sessions=deepcopy(intent['sessions']))
                if case == 'hash':
                    plan['intent_sha256'] = '0'*64
                elif case == 'count':
                    plan['sessions'] = []
                else:
                    plan['sessions'][0]['start_updates'] = 1
                (root/'execution-plan.json').write_text(json.dumps(plan))
                for flags in ([], ['-O']):
                    result = subprocess.run([sys.executable, *flags, str(script), str(root)],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Frozen intent ' + message, result.stderr)


class ExecutionEvidenceTests(unittest.TestCase):
    def setUp(self):
        import io
        import zipfile
        from scripts.protocol_gate import digest
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as z:
            z.writestr('dougpu/train.py', b'# synthetic trainer')
        self.snapshot = out.getvalue()
        source = dict(snapshot_sha256=digest(self.snapshot),
                      files={'dougpu/train.py': dict(path='/fixture/dougpu/train.py',
                             sha256=digest(b'# synthetic trainer'))},
                      loaded_modules={'dougpu.train': 'dougpu/train.py'})
        self.rows, sessions = [], []
        for index, (begin, end) in enumerate(((0, 8), (8, 16))):
            tc = asdict(TrainConfig(workers=1, target_updates=end))
            common = dict(session_id=str(index), updates=begin)
            resume = dict(path='/fixture/ckpt.zip', sha256='a'*64, updates=begin, cycle=1) if index else None
            actor = dict(worker_seeds=[42 + index], worker_order=[0], mode='ordered', packed=True, **common)
            versions = dict(python='fixture', jax='fixture', numpy='fixture', backend='cpu')
            start = dict(versions=versions, model=asdict(ModelConfig()), train=tc, source_lock={'engine': 'fixture'},
                         runtime_source=deepcopy(source), resume_input=resume, **common)
            self.rows.extend([dict(event='actor_start', **actor), dict(event='start', **start),
                              dict(event='train', session_id=str(index), updates=end, successful_steps=end-begin)])
            endpoint = dict(target_updates=end, status='COMPLETE', stop_reason='target_updates')
            self.rows.append(dict(event='session_end', session_id=str(index), updates=end,
                                  stop_reason='target_updates', update_endpoint=endpoint))
            sessions.append(dict(versions=versions, start_updates=begin, end_updates=end,
                model=start['model'], train=tc, source_lock=start['source_lock'],
                source_sha256=source['snapshot_sha256'], input_checkpoint_sha256='a'*64 if index else None))
        endpoint = dict(target_updates=16, status='COMPLETE', stop_reason='target_updates')
        self.meta = dict(start, updates=16, actor_start=actor, reason='session_end', update_endpoint=endpoint)
        self.plan = dict(sessions=sessions)

    def test_execution_positive_and_negative(self):
        from scripts.protocol_gate import execution_sessions
        self.assertEqual(len(execution_sessions(self.rows, self.meta, self.snapshot, self.plan)), 2)
        for change in ('seed', 'resume', 'source', 'boundary', 'forged_status', 'actor_metadata', 'null_actor', 'missing_middle_end', 'incomplete_middle', 'version'):
            rows, meta, plan = deepcopy(self.rows), deepcopy(self.meta), deepcopy(self.plan)
            if change == 'missing_middle_end':
                rows.pop(3)
            elif change == 'incomplete_middle':
                rows[3]['update_endpoint']['status'] = 'INCOMPLETE'
            elif change == 'version':
                rows[1]['versions']['jax'] = 'other'
            elif change == 'seed':
                del rows[4]['worker_seeds']
            elif change == 'resume':
                rows[5]['resume_input']['sha256'] = 'b'*64
            elif change == 'source':
                rows[5]['runtime_source']['files']['dougpu/train.py']['sha256'] = 'b'*64
            elif change == 'boundary':
                plan['sessions'].pop(0)
            elif change == 'forged_status':
                rows[5]['runtime_source'] = 'ATTESTED'
            elif change == 'null_actor':
                meta['actor_start'] = None
            else:
                meta['actor_start'] = {}
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError, TypeError)):
                execution_sessions(rows, meta, self.snapshot, plan)


if __name__ == '__main__':
    unittest.main()
