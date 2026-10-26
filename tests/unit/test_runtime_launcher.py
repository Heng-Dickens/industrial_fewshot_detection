from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main as entrypoint


class RuntimeLauncherTests(unittest.TestCase):
    def test_incomplete_interpreter_uses_verified_project_runtime(self):
        with patch.object(entrypoint.sys, 'executable', 'C:/missing/base_python.exe'), \
             patch.object(entrypoint.Path, 'is_file', return_value=True), \
             patch.object(entrypoint.importlib.util, 'find_spec', return_value=None), \
             patch.object(entrypoint.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as check:
            runtime = entrypoint.project_runtime(['--config', 'experiment.yaml'])
        self.assertTrue(str(runtime).endswith('python.exe'))
        self.assertEqual(check.call_count, 1)

    def test_working_interpreter_is_preserved(self):
        with patch.object(entrypoint.sys, 'executable', 'C:/working/python.exe'), \
             patch.object(entrypoint.Path, 'is_file', return_value=True), \
             patch.object(entrypoint.importlib.util, 'find_spec', return_value=object()), \
             patch.object(entrypoint.subprocess, 'run') as check:
            self.assertIsNone(entrypoint.project_runtime(['--config', 'experiment.yaml']))
        check.assert_not_called()

    def test_planning_evaluation_and_explicit_override_do_not_switch(self):
        for flags in (['--config', 'x', '--dry-run'], ['--evaluate-only', 'run'], ['--help']):
            self.assertIsNone(entrypoint.project_runtime(flags))
        with patch.dict(entrypoint.os.environ, {'INDUSTRIAL_USE_CURRENT_PYTHON': '1'}):
            self.assertIsNone(entrypoint.project_runtime(['--config', 'x']))
