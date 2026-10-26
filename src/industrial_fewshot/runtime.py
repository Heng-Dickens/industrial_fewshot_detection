"""Responsive console interruption and cleanup of explicitly owned processes."""
import asyncio
import os
import signal
import subprocess
import sys
import threading


def run_async(awaitable):
    state = {'interrupted': False, 'loop': None, 'task': None}
    previous = {}

    def interrupt(signum, frame):
        if state['interrupted']:
            print('再次中断，立即退出；已经保存的结果保留。', file=sys.stderr, flush=True)
            os._exit(130)
        state['interrupted'] = True
        if state['loop'] is not None:
            state['loop'].call_soon_threadsafe(state['task'].cancel)
        else:
            raise KeyboardInterrupt

    async def heartbeat():
        # Python 3.10 on Windows needs bounded IO-loop polls for console signals.
        while True:
            await asyncio.sleep(0.1)

    async def managed():
        state.update(loop=asyncio.get_running_loop(), task=asyncio.current_task())
        pulse = asyncio.create_task(heartbeat())
        try:
            return await awaitable
        finally:
            pulse.cancel()
            await asyncio.gather(pulse, return_exceptions=True)

    if threading.current_thread() is threading.main_thread():
        for name in ('SIGINT', 'SIGBREAK'):
            signum = getattr(signal, name, None)
            if signum is not None:
                previous[signum] = signal.signal(signum, interrupt)
    try:
        return asyncio.run(managed())
    except asyncio.CancelledError:
        if state['interrupted']:
            raise KeyboardInterrupt from None
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def stop_owned_process(process, grace_seconds=5):
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGINT)
        process.wait(timeout=grace_seconds)
        return
    except (OSError, ValueError, KeyboardInterrupt, subprocess.TimeoutExpired):
        pass
    if os.name == 'nt' and process.poll() is None:
        # Only the process created by this caller and its children are targeted.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=False)
    if process.poll() is None:
        process.kill()
    process.wait(timeout=grace_seconds)


def run_project_child(runtime, script, argv):
    process = subprocess.Popen([str(runtime), str(script), *argv],
                              creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
    previous = None
    if hasattr(signal, 'SIGBREAK'):
        previous = signal.signal(signal.SIGBREAK, signal.default_int_handler)
    try:
        while True:
            try:
                return process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        stop_owned_process(process)
        return 130
    finally:
        if previous is not None:
            signal.signal(signal.SIGBREAK, previous)
