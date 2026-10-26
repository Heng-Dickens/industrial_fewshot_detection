"""Claude Messages transport through the existing project SDK HTTP client."""
from __future__ import annotations

import re

from industrial_fewshot.service_errors import ModelServiceError


def _check_model(response, expected, require_present=False):
    actual = response.get('model')
    if require_present and not actual:
        raise ModelServiceError('代理响应缺少实际模型名称，无法核验路由别名对应的模型，已停止实验')
    if expected and actual and actual != expected:
        raise ModelServiceError(f'代理返回模型 {actual} 与请求模型 {expected} 不一致，已停止实验')


def message_body(config: dict, messages: list, response_format: dict | None = None) -> dict:
    converted = []
    for message in messages:
        blocks = []
        content = message['content']
        for block in ([{'type': 'text', 'text': content}] if isinstance(content, str) else content):
            if block['type'] == 'text':
                blocks.append(dict(block))
            elif block['type'] == 'image_url':
                url = block['image_url']['url']
                match = re.fullmatch(r'data:(image/[\w.+-]+);base64,(.+)', url, re.DOTALL)
                if not match:
                    raise ValueError('Claude 图片输入必须是 base64 data URL')
                blocks.append({'type': 'image', 'source': {
                    'type': 'base64', 'media_type': match[1], 'data': match[2]}})
            else:
                raise ValueError('Claude 消息包含不支持的内容类型')
        converted.append({'role': message['role'], 'content': blocks})
    body = {key: config[key] for key in ('model', 'temperature', 'top_p', 'max_tokens', 'stream')}
    # A local proxy routing alias is distinct from the required upstream model.
    # read_message still validates the response against config['model'].
    body['model'] = config.get('proxy_request_model', config['model'])
    body['messages'] = converted
    body['thinking'] = {'type': 'disabled'}
    if config.get('enable_thinking'):
        raise ValueError('当前 Claude 转发适配请使用 enable_thinking: false')
    if response_format:
        if response_format['type'] == 'json_schema':
            body['output_config'] = {'format': {
                'type': 'json_schema', 'schema': response_format['json_schema']['schema']}}
        elif response_format['type'] == 'json_object':
            body['system'] = '只输出一个合法的 JSON 对象，不要 Markdown 代码块或额外说明。'
        else:
            raise ValueError('Claude 转发不支持该结构化输出格式')
    return body


def _finish_reason(reason):
    return {'end_turn': 'stop', 'stop_sequence': 'stop', 'max_tokens': 'length',
            'tool_use': 'tool_calls'}.get(reason, reason)


async def read_message(response, stream: bool, expected_model=None, require_model=False):
    """Keep answer text and usage; do not expose private thinking blocks."""
    if not stream:
        if not isinstance(response, dict) or not isinstance(response.get('content'), list):
            raise ValueError('Claude 模型响应没有 content 数组')
        _check_model(response, expected_model, require_model)
        content = ''.join(block.get('text', '') for block in response['content']
                          if block.get('type') == 'text')
        return content, _finish_reason(response.get('stop_reason')), response.get('usage'), None
    parts, usage, reason = [], {}, None
    stopped = False
    try:
        async for event in response:
            kind = event.get('type')
            if kind == 'message_start':
                _check_model(event.get('message', {}), expected_model, require_model)
                usage.update(event.get('message', {}).get('usage') or {})
            elif kind == 'content_block_start':
                block = event.get('content_block', {})
                if block.get('type') == 'text':
                    parts.append(block.get('text', ''))
            elif kind == 'content_block_delta':
                delta = event.get('delta', {})
                if delta.get('type') == 'text_delta':
                    parts.append(delta.get('text', ''))
            elif kind == 'message_delta':
                reason = event.get('delta', {}).get('stop_reason', reason)
                usage.update(event.get('usage') or {})
            elif kind == 'message_stop':
                stopped = True
            elif kind == 'error':
                raise RuntimeError('Claude 流式请求返回错误事件')
    finally:
        if hasattr(response, 'close'):
            await response.close()
    if not stopped:
        raise ValueError('Claude 流式响应中断，缺少 message_stop')
    return ''.join(parts), _finish_reason(reason), usage or None, None


async def complete_message(client, config: dict, messages: list, response_format=None):
    body = message_body(config, messages, response_format)
    kwargs = {'cast_to': object, 'body': body,
              'options': {'headers': {'anthropic-version': '2023-06-01'}}}
    if config['stream']:
        from openai import AsyncStream
        kwargs.update(stream=True, stream_cls=AsyncStream[object])
    response = await client.post('/messages', **kwargs)
    return await read_message(response, config['stream'], config['model'], bool(config.get('proxy_request_model')))
