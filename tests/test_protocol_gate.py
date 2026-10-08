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


if __name__ == '__main__':
    unittest.main()
