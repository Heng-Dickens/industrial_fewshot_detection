import asyncio
import os
import signal
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from industrial_fewshot.runtime import run_async, run_project_child, stop_owned_process


class RuntimeShutdownTest(unittest.TestCase):
    def test_real_sigint_cancels_hanging_request_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGINT)
        state = {'closed': False}
        async def hanging():
            asyncio.get_running_loop().call_later(0.05, signal.raise_signal, signal.SIGINT)
            try:
                await asyncio.Event().wait()
            finally:
                state['closed'] = True
        with self.assertRaises(KeyboardInterrupt):
            run_async(hanging())
        self.assertTrue(state['closed'])
        self.assertIs(signal.getsignal(signal.SIGINT), previous)

    def test_programmatic_cancellation_is_not_misreported_as_ctrl_c(self):
        async def cancelled():
            raise asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            run_async(cancelled())

    def test_managed_runtime_child_exit_code_is_preserved(self):
        self.assertEqual(run_project_child(sys.executable, '-c', ['raise SystemExit(7)']), 7)

    def test_interrupt_is_forwarded_to_the_owned_runtime_child(self):
        child = MagicMock()
        child.wait.side_effect = [KeyboardInterrupt, 0]
        child.poll.return_value = None
        with patch('industrial_fewshot.runtime.subprocess.Popen', return_value=child):
            self.assertEqual(run_project_child(sys.executable, 'main.py', ['--config', 'x.yaml']), 130)
        child.send_signal.assert_called_once_with(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGINT)
        child.kill.assert_not_called()

    def test_actual_owned_sleeping_child_is_stopped(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                 creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
        try:
            stop_owned_process(child, grace_seconds=0.3)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=3)
