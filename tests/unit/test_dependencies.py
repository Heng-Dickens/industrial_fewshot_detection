import builtins
import sys
import unittest
from unittest.mock import patch

from industrial_fewshot.dependencies import load_openai_client


class DependencyErrorTests(unittest.TestCase):
    def test_missing_sdk_names_interpreter_and_original_error(self):
        original_import = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == 'openai':
                raise ModuleNotFoundError("No module named 'openai'", name='openai')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=missing):
            with self.assertRaises(RuntimeError) as caught:
                load_openai_client()
        self.assertIn('当前解释器未安装 openai', str(caught.exception))
        self.assertIn(sys.executable, str(caught.exception))
        self.assertIn('ModuleNotFoundError', str(caught.exception))

    def test_broken_dependency_is_not_reported_as_missing_sdk(self):
        original_import = builtins.__import__
        def broken(name, *args, **kwargs):
            if name == 'openai':
                raise ImportError('DLL load failed while importing jiter')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=broken):
            with self.assertRaises(RuntimeError) as caught:
                load_openai_client()
        self.assertIn('openai 或其依赖导入失败', str(caught.exception))
        self.assertIn('DLL load failed while importing jiter', str(caught.exception))
        self.assertNotIn('未安装 openai', str(caught.exception))
