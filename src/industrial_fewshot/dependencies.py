"""Dependency errors must identify the actual interpreter and original cause."""
import os
from pathlib import Path
import sys


def load_openai_client():
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        missing = isinstance(exc, ModuleNotFoundError) and exc.name == 'openai'
        problem = '当前解释器未安装 openai' if missing else 'openai 或其依赖导入失败'
        requirements = Path(__file__).resolve().parents[2] / 'requirements.txt'
        prefix = '& ' if os.name == 'nt' else ''
        command = f'{prefix}"{sys.executable}" -m pip install -r "{requirements}"'
        raise RuntimeError(f'{problem}。Python: {sys.executable}。'
                           f'原始错误: {type(exc).__name__}: {exc}。'
                           f'在同一解释器中检查/补齐依赖: {command}') from exc
    return AsyncOpenAI
