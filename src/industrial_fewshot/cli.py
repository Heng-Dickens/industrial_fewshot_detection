"""Command-line entrypoint for the experiment framework."""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"prepare": ".data.preprocessing", "compare": ".evaluation.compare",
                "stability": ".evaluation.fewshot_stability", "visualize": ".evaluation.visualize",
                "prepare-references": ".data.prepare"}
    if argv and argv[0] in commands:
        import importlib
        return importlib.import_module(commands[argv[0]], __package__).main(argv[1:])
    if argv and argv[0] == "run":
        argv.pop(0)
    if argv and argv[0] == "evaluate":
        argv = argv[1:] if argv[1:] in (["--help"], ["-h"]) else ["--evaluate-only", *argv[1:]]
    from .runner import run
    from .config import load_config
    from .evaluation.evaluator import evaluate_saved_run
    from .runtime import run_async
    parser = argparse.ArgumentParser(description="统一目标标注；train=标注+评估，test=仅标注")
    parser.add_argument("--config", type=Path, help="本次场景的独立配置文件；只评估时可用于指定评估参数")
    parser.add_argument("--evaluate-only", type=Path, metavar="RUN_DIR", help="只读取已有运行目录补算指标，不调用模型、不重新推理")
    parser.add_argument("--mode", choices=["train", "test"])
    parser.add_argument("--image-dir", type=Path, help="覆盖待标注图片目录")
    parser.add_argument("--output-dir", type=Path, help="覆盖配置中的输出根目录；两者均未指定时使用输入目录名加后缀的同级目录")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置/数据/参考，不调用模型、不写结果")
    parser.add_argument("--limit", type=int, help="调试时限制图片数；默认扫描整个目录")
    parser.add_argument("--force", action="store_true", help="明确忽略成功结果缓存，重新推理（远程模型会消耗 token）")
    parser.add_argument("--ablation", action="store_true", help="按配置依次运行参考输入对比实验，输出各组数值汇总；需 train 模式")
    parser.add_argument('--report-format', choices=['text', 'json'], help='终端报告格式；默认表格，dry-run 默认 JSON')
    parser.add_argument('--json', dest='report_format', action='store_const', const='json', help='终端输出完整 JSON，供程序读取')
    args = parser.parse_args(argv)
    if args.evaluate_only is None and args.config is None:
        parser.error("执行标注必须指定 --config 场景配置；补算指标请指定 --evaluate-only 运行目录")
    if args.evaluate_only is not None and (args.mode is not None or args.image_dir is not None or args.output_dir is not None or args.limit is not None or args.force or args.ablation):
        parser.error("--evaluate-only 不能同时指定 --mode、--image-dir、--output-dir、--limit、--force 或 --ablation")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.evaluate_only is not None:
            report = evaluate_saved_run(args.evaluate_only, args.config, args.dry_run)
        else:
            config = load_config(args.config, args.mode, args.image_dir, args.output_dir)
            LOGGER.info("实际 Python: %s", sys.executable)
            LOGGER.info("配置文件: %s", config['config_file'])
            LOGGER.info("评估参数: IoU=%s | score_threshold=%s | maxDets=%s",
                        config['evaluation']['iou_threshold'], config['evaluation']['score_threshold'],
                        config['evaluation']['max_detections_per_image'])
            for target in config['targets']:
                LOGGER.info("实际模型: %s | 服务: %s", target['config']['model'],
                            target['config'].get('base_url') or '本地推理')
                if target['backend'] == 'groundingdino':
                    LOGGER.info('实际文字 prompt: %s', target['config']['groundingdino']['text_prompt'])
                if target['backend'] == 'yoloe_fusion':
                    settings = target['config']['yoloe_fusion']
                    LOGGER.info('YOLOE 联合提示: %sshot | 裁剪视觉权重=%s | 文字权重=%s | seed=%s',
                                settings['positive_count'], settings['visual_weight'], settings['text_weight'], settings['selection_seed'])
                    LOGGER.info('实际文字 prompt: %s', settings['text_prompt'])
                if target['backend'] == 'yoloe_text':
                    settings = target['config']['yoloe_text']
                    LOGGER.info('YOLOE 纯文字提示 | 0shot | 参考图输入=0 | imgsz=%s | confidence=%s | NMS IoU=%s',
                                settings['imgsz'], settings['confidence'], settings['nms_iou'])
                    LOGGER.info('实际文字 prompt: %s', settings['text_prompt'])
                    LOGGER.info('文字编码器: %s', settings['text_encoder_weights'])
                if target['backend'] == 'yolo_onnx':
                    settings = target['config']['yolo_onnx']
                    LOGGER.info('监督检测 ONNX: 输入=%s（高×宽） | confidence=%s | NMS IoU=%s | 类别=%s | providers=%s',
                                settings['input_size'], settings['confidence'], settings['nms_iou'],
                                settings['class_names'], settings['providers'])
                if target['backend'] == 'yoloe':
                    settings = target['config']['yoloe']
                    LOGGER.info("YOLOE 推理参数: shot=%s | seed=%s | confidence=%s | imgsz=%s | NMS IoU=%s",
                                settings['positive_count'], settings['selection_seed'], settings['confidence'],
                                settings['imgsz'], settings['nms_iou'])
                if target['config'].get('base_url'):
                    protocol = target['config'].get('api_protocol', 'openai_chat_completions')
                    endpoint = '/messages（CC Switch Claude 分组）' if protocol == 'anthropic_messages' else '/chat/completions'
                    LOGGER.info('调用协议: %s | 请求接口: %s%s', protocol,
                                target['config']['base_url'].rstrip('/'), endpoint)
            if args.ablation or config["ablation"]["enabled"] or config["experiment_suite"]["enabled"]:
                from industrial_fewshot.experiments import run_ablation
                report = run_async(run_ablation(config, args.dry_run, args.limit, force=args.force))
            else:
                report = run_async(run(config, args.dry_run, args.limit, force=args.force))
        output_format = args.report_format or ('json' if args.dry_run else 'text')
        if output_format == 'json':
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            from .evaluation.reporting import format_report
            print(format_report(report))
        return 2 if report.get("failed_request_count", 0) else 0
    except KeyboardInterrupt:
        LOGGER.info('已停止（Ctrl+C）；已经保存的结果和成功缓存保留。')
        return 130
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        LOGGER.error("程序退出: %s", exc)
        return 1

