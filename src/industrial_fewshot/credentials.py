"""Resolve project-scoped API credentials without changing desktop app settings."""
import json
import os
import re
from pathlib import Path


def _environment_file_key(path: Path, name: str) -> str:
    """Read a literal KEY=value file, without expansion or executing shell code."""
    try:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f'无法读取环境变量文件: {path} ({type(exc).__name__})') from None
    selected = None
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)', line)
        if not match:
            raise RuntimeError(f'环境变量文件格式错误: {path} 第 {number} 行；请使用 KEY=value')
        variable, value = match.groups()
        if value.startswith(('"', "'")):
            quoted = re.fullmatch(r"([\"'])(.*?)\1\s*(?:#.*)?", value)
            if quoted is None:
                raise RuntimeError(f'环境变量文件引号不完整: {path} 第 {number} 行')
            value = quoted.group(2)
        else:
            value = re.split(r'\s+#', value, maxsplit=1)[0].strip()
        if variable == name:
            selected = value
    if not selected or not selected.strip() or selected.strip() == 'EMPTY':
        raise RuntimeError(f'请在环境变量文件 {path} 中填写 {name}，或设置同名进程环境变量')
    return selected.strip()


def resolve_api_key(model: dict) -> str:
    environment_name = model.get('api_key_env', 'VLLM_API_KEY')
    environment_key = os.environ.get(environment_name)
    if environment_key and environment_key.strip():
        return environment_key.strip()
    if model.get('env_file'):
        return _environment_file_key(Path(model['env_file']).expanduser(), environment_name)
    if model.get('api_key_file'):
        path = Path(model['api_key_file']).expanduser()
        try:
            value = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f'无法读取实验专用密钥文件: {path}（{type(exc).__name__}）。'
                f'请配置该 JSON 文件中的 api_key，或设置环境变量 {environment_name}。'
            ) from None
        key = value.get('api_key') if isinstance(value, dict) else None
        if not isinstance(key, str) or not key.strip() or key.strip() == 'EMPTY':
            raise RuntimeError(f'实验专用密钥文件必须包含非空 api_key: {path}')
        return key.strip()
    # Preserve flat legacy configurations and keyless local compatible servers.
    return model.get('api_key', 'EMPTY')
