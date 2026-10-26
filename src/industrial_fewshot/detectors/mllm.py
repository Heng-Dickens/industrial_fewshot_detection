"""One model-request implementation for all configured target strategies.

Protocol parsing, streaming and schema validation reuse tool.batch_annotate;
this module adds reference messages and optional suspension filtering.
"""
from __future__ import annotations

import logging
import asyncio
import hashlib
from industrial_fewshot.config import render_prompt
from industrial_fewshot.labels import target_label_names
import industrial_fewshot.prompts.references as references
from typing import Any

from industrial_fewshot.postprocess.schema import build_response_format, clean_json_response, validate_prediction
from industrial_fewshot.detectors.transport import get_image_info, image_mime_type, read_completion
from industrial_fewshot.prompts.references import build_messages
from industrial_fewshot.service_errors import ModelServiceError, connection_failure, terminal_service_error

LOGGER = logging.getLogger(__name__)


class ModelOutputError(ValueError):
    """Keep a returned model answer available when its JSON/schema is invalid."""
    def __init__(self, message, model_output):
        super().__init__(message)
        self.model_output = model_output


def apply_suspension_gate(
    prediction: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """丢弃“框内没有已离地吊物本体”的候选框。

    在 validate_prediction 之前对模型原始响应执行，因此必须容忍畸形输入：
    objects 不是数组、或元素不是 object 时原样返回，交给 validate_prediction 报正常错误。

    返回 (过滤后的响应, 统计信息)。统计信息只用于检查点留痕，不进入 objects。
    """
    info = {
        "enabled": bool(config.get("require_suspension_evidence")),
        "dropped": 0,
        "missing_flag": 0,
    }
    if not info["enabled"]:
        return prediction, info

    objects = prediction.get("objects")
    if not isinstance(objects, list) or not all(
        isinstance(item, dict) for item in objects
    ):
        return prediction, info

    kept: list[dict[str, Any]] = []
    missing = 0
    for item in objects:
        flag = item.get("hanging_object_confirmed")
        if flag is False:
            continue
        if not isinstance(flag, bool):
            # 字段缺失或不是布尔时保留：服务端忽略 schema 时不能让整批结果清成无目标
            missing += 1
        kept.append(item)

    dropped = len(objects) - len(kept)
    info["missing_flag"] = missing
    info["dropped"] = dropped
    if not dropped:
        # 没有丢弃时不动 target_count，保留“数量与 objects 不一致”这个有效报错信号
        return prediction, info

    if missing:
        LOGGER.warning(
            "有 %d 个目标缺少 hanging_object_confirmed 字段，已按通过处理；"
            "请确认服务端是否真的支持 json_schema 结构化输出",
            missing,
        )

    gated = dict(prediction)
    gated["objects"] = kept
    gated["target_count"] = len(kept)
    if not kept and gated.get("annotation_status") == "annotated":
        gated["annotation_status"] = "no_target"
        gated["reason_code"] = "no_target"
        note = (
            f"已丢弃 {dropped} 个框内看不到被吊起吊物本体的候选"
            "（只框住吊具/吊索，或吊物仍未离开支撑面）。"
        )
        reason = str(gated.get("reason") or "").strip()
        gated["reason"] = f"{reason} {note}".strip()
    return gated, info



async def request_annotation(
    client: Any,
    config: dict[str, Any],
    prompt: str,
    reference_examples: list[dict[str, Any]],
    current_url: str,
    response_format: dict[str, Any] | None,
    label_names: set[str],
    width: int,
    height: int,
) -> dict[str, Any]:
    """发送一次带参考示例的结构化标注请求并完成业务校验。"""
    message_content = build_messages(config, prompt, reference_examples, current_url)
    request: dict[str, Any] = {
        "model": config["model"],
        "messages": [{"role": "user", "content": message_content}],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_tokens"],
        "stream": config["stream"],
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": config["enable_thinking"],
            }
        },
    }
    if config["stream"]:
        request["stream_options"] = {"include_usage": True}
    if response_format is not None:
        request["response_format"] = response_format

    if config.get('api_protocol') == 'anthropic_messages':
        from .claude_messages import complete_message
        content, finish_reason, usage_dict, reasoning_content = await complete_message(
            client, config, request['messages'], response_format)
    else:
        response = await client.chat.completions.create(**request)
        content, finish_reason, usage_dict, reasoning_content = await read_completion(
            response, config["stream"]
        )
    output = {'raw_response': content, 'finish_reason': finish_reason, 'usage': usage_dict}
    if not content.strip():
        reasoning_length = len(reasoning_content or "")
        raise ModelOutputError(
            "模型响应内容为空"
            f"（finish_reason={finish_reason!r}, "
            f"reasoning_content字符数={reasoning_length}）", output
        )
    try:
        prediction = clean_json_response(content)
        prediction, suspension_gate = apply_suspension_gate(prediction, config)
        annotation_result = validate_prediction(prediction, label_names, width, height)
        if config.get('require_box_reason') and any(not obj.get('reason') for obj in annotation_result['objects']):
            raise ValueError('模型未返回逐框 reason，请提供可见依据和框覆盖范围')
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise ModelOutputError(str(exc), output) from exc
    return {
        "annotation": annotation_result,
        "suspension_gate": suspension_gate,
        "raw_response": content,
        "reasoning_content": reasoning_content,
        "finish_reason": finish_reason,
        "usage": usage_dict,
    }


async def infer_image(client, target: dict, item: dict) -> dict:
    config = target["config"]
    record = {"relative_path": item["relative_path"], "target_id": target["id"], "status": "failed"}
    feedback_attempts = []
    try:
        data = await asyncio.to_thread(item["path"].read_bytes)
        checksum = hashlib.sha256(data).hexdigest()
        if item.get("image_sha256") and checksum != item["image_sha256"]:
            raise ValueError("图片在检查后发生变化，请重试本次运行")
        record["image_sha256"] = checksum
        _, _, image_format = get_image_info(data)
        url = references.encode_data_url(data, image_mime_type(item["path"], image_format))
        response_format = build_response_format(config)

        async def request(prompt: str, stage: str) -> dict:
            result = None
            event = {'attempt': attempt, 'stage': stage}
            try:
                result = await request_annotation(
                    client, config, prompt, config['reference_examples'], url,
                    response_format, set(target_label_names(target)), item['width'], item['height'],
                )
                if any('confidence' not in obj for obj in result['annotation']['objects']):
                    raise ModelOutputError('模型未返回 confidence，不能可靠计算排序 AP', result)
                event.update(status='success')
                return result
            except Exception as exc:
                result = getattr(exc, 'model_output', None)
                event.update(status='invalid_response' if result is not None else 'request_failed',
                             error=str(terminal_service_error(exc) or exc))
                raise
            finally:
                if result is not None:
                    event.update({key: result[key] for key in ('raw_response', 'finish_reason', 'usage', 'annotation', 'suspension_gate') if key in result})
                feedback_attempts.append(event)

        for attempt in range(1, config["max_attempts"] + 1):
            try:
                initial = await request(target["prompt"], 'initial')
                final = initial
                review_performed, review_error = False, None
                if config["enable_completeness_review"]:
                    review_prompt = target["prompt"] + "\n\n" + render_prompt(target["review_template"], target, initial["annotation"])
                    try:
                        final = await request(review_prompt, 'review')
                        review_performed = True
                    except Exception as exc:
                        fatal = terminal_service_error(exc)
                        if fatal:
                            raise fatal from exc
                        review_error = str(exc)
                return {
                    **record, "status": "success", **final["annotation"], "attempts": attempt,
                    "initial_raw_response": initial["raw_response"], "raw_response": final["raw_response"],
                    'initial_annotation': initial['annotation'],
                    'initial_suspension_gate': initial['suspension_gate'], 'suspension_gate': final['suspension_gate'],
                    'initial_finish_reason': initial['finish_reason'], 'finish_reason': final['finish_reason'],
                    "initial_usage": initial["usage"], "usage": final["usage"],
                    "review_performed": review_performed, "review_error": review_error,
                    'model_feedback_attempts': feedback_attempts,
                }
            except ModelServiceError:
                raise
            except Exception as exc:
                fatal = terminal_service_error(exc)
                if fatal:
                    raise fatal from exc
                if attempt >= config["max_attempts"]:
                    if connection_failure(exc):
                        raise ModelServiceError(
                            f'模型服务连续 {attempt} 次连接失败或超时，已停止本次实验；请检查代理和模型服务。'
                        ) from exc
                    raise
                delay = config['retry_delay_seconds'] * 2 ** (attempt - 1)
                LOGGER.warning('请求失败 %s/%s，第 %d/%d 次（%s）；%s 秒后重试，可按 Ctrl+C 停止。',
                               target['id'], item['relative_path'], attempt, config['max_attempts'],
                               type(exc).__name__, delay)
                await asyncio.sleep(delay)
    except ModelServiceError as exc:
        exc.record = {**record, 'error': str(exc), 'model_feedback_attempts': feedback_attempts}
        raise
    except Exception as exc:
        LOGGER.error("标注失败 %s/%s: %s", target["id"], item["relative_path"], exc)
        return {**record, "error": str(exc), 'model_feedback_attempts': feedback_attempts}


class MLLMDetector:
    def __init__(self, client, target):
        self.client, self.target = client, target

    async def predict(self, image):
        return await infer_image(self.client, self.target, image.as_item())

    async def close(self):
        await self.client.close()


