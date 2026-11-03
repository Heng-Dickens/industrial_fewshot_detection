"""Describe SAVPE behavior from the actual training method, never a run's name."""
import json
import os
from collections import Counter
from pathlib import Path


def method_policy(method, scope=None, *, freeze_savpe_bn_stats=False, pretrained=False, freeze_savpe=False):
    if freeze_savpe and (method != 'yoloe-vp' or pretrained):
        raise ValueError('Frozen SAVPE applies only to visual-prompt fine-tuning.')
    if method == 'yoloe-head':
        scope = scope or 'head'
        ranges = {'head':'仅融合后的分类输出层','neck':'neck + 固定检测头','full':'backbone + neck + 固定检测头'}
        if scope not in ranges:
            raise ValueError('Invalid fixed-head scope.')
        return dict(method_label='YOLOE固定头（不保留SAVPE）',savpe_state='removed',savpe_present=False,
            savpe_finetuned=False,uses_savpe_at_inference=False,savpe_bn_statistics='not_applicable',
            training_range=ranges[scope],inference_mode='fixed_head')
    if method == 'yoloe-vp':
        scope = scope or 'savpe'
        ranges = {'savpe':'仅SAVPE','head':'检测头（含SAVPE）','neck':'neck + 检测头 + SAVPE',
                  'full':'backbone + neck + 检测头 + SAVPE'}
        if scope not in ranges:
            raise ValueError('Invalid visual-prompt scope.')
        if freeze_savpe and scope == 'savpe':
            raise ValueError('No trainable parameters after freezing SAVPE-only scope.')
        if freeze_savpe:
            ranges = {'head':'检测头（SAVPE冻结）','neck':'neck + 检测头（SAVPE冻结）',
                      'full':'backbone + neck + 检测头（SAVPE冻结）'}
        return dict(method_label='原始YOLOE视觉提示（未微调）' if pretrained else 'YOLOE视觉提示（保留并冻结SAVPE）' if freeze_savpe else 'YOLOE视觉提示（保留并微调SAVPE）',
            savpe_state='retained',savpe_present=True,savpe_finetuned=not pretrained and not freeze_savpe,
            savpe_parameters_frozen=freeze_savpe,
            uses_savpe_at_inference=True,
            savpe_bn_statistics='not_trained' if pretrained else 'frozen' if freeze_savpe_bn_stats or freeze_savpe else 'updated',
            training_range='未微调' if pretrained else ranges[scope],inference_mode='visual_prompt')
    if method == 'yolov8':
        return dict(method_label='YOLOv8监督基线（原结构无SAVPE）',savpe_state='not_applicable',savpe_present=False,
            savpe_finetuned=False,uses_savpe_at_inference=False,savpe_bn_statistics='not_applicable',
            training_range='backbone + neck + 普通检测头',inference_mode='fixed_head')
    raise ValueError(f'Unknown fine-tuning method: {method}')


def metadata_policy(metadata, run_dir):
    adaptation = metadata.get('adaptation', {})
    policy = method_policy(metadata['method'], metadata['strategy'],
        freeze_savpe_bn_stats=adaptation.get('freeze_savpe_bn_stats', False),
        freeze_savpe=adaptation.get('freeze_savpe', False))
    parameters = Path(run_dir)/'trainable_parameters.json'
    if parameters.exists():
        names = json.loads(parameters.read_text(encoding='utf-8-sig'))
        if any('.savpe.' in name for name in names) != policy['savpe_finetuned']:
            raise ValueError(f'SAVPE parameter evidence contradicts method: {run_dir}')
    return policy


def verify_checkpoint_policy(policy, evidence):
    structure = evidence.get('checkpoint_structure', {})
    if 'has_savpe' in structure and structure['has_savpe'] != policy['savpe_present']:
        raise ValueError('Checkpoint SAVPE presence contradicts training metadata.')
    if policy['savpe_state'] in {'removed','retained'} and 'is_fused' in structure:
        if structure['is_fused'] != (policy['savpe_state'] == 'removed'):
            raise ValueError('Checkpoint fusion state contradicts training metadata.')


def write_method_document(report, rows, project_root):
    guide = os.path.relpath(Path(project_root)/'docs/guides/YOLOE_FINETUNING_METHODS.md',report)
    counts = Counter(r['savpe_state'] for r in rows)
    lines = ['# 逐实验SAVPE使用情况', '',
        '按实际训练元数据分类，并核验可训练参数清单与可用的推理结构记录；不根据目录名称猜测。',
        f'共{len(rows)}项训练：移除SAVPE {counts["removed"]}项，保留SAVPE {counts["retained"]}项（其中冻结参数{sum(bool(r.get("savpe_parameters_frozen")) for r in rows)}项），原结构无SAVPE {counts["not_applicable"]}项。',
        f'完整机制和复现说明见[微调方法指南]({guide})；可筛选[逐实验CSV](SAVPE_METHODS.csv)。', '',
        '“推理使用”表示用当前模型的SAVPE编码参考图片，再把类别原型用于query检测；支持原型可缓存。',
        '仅冻结BN统计时卷积与BN仿射参数仍可训练；freeze_savpe=true同时冻结全部SAVPE参数及统计。所有full范围仍遵守上游DFL固定投影等例外。', '',
        '| 实验 | 方法 | 训练范围 | 推理使用SAVPE | BN统计 | 协议/损失 | 训练seed | 训练尺寸 | 证据 |',
        '|---|---|---|---|---|---|---:|---:|---|']
    for row in sorted(rows,key=lambda r:(r['method'],r['scope'],r['id'])):
        trial = 'trials/'+row['id']
        lines.append(f'| {row["id"]} | {row["method_label"]} | {row["training_range"]} | '
            f'{"是" if row["uses_savpe_at_inference"] else "否"} | {row["savpe_bn_statistics"]} | '
            f'{row["protocol"]}/{row["loss"]} | {row["training_seed"]} | {row["imgsz"]} | '
            f'[元数据]({trial}/finetune_config.json) / [训练参数]({trial}/trainable_parameters.json) |')
    lines += ['', '训练均使用10shot；VP测试的5/10shot是推理参考数量。固定头和YOLOv8没有视觉shot维度。', '']
    if (Path(report)/'frozen_savpe_20261030/plan.json').exists():
        lines += ['另有[冻结整个SAVPE的新增条件](frozen_savpe_20261030/README.md)，合并结果见[全部微调总表](ALL_EXPERIMENTS.md)。','']
    (Path(report)/'SAVPE_METHODS.md').write_text('\n'.join(lines),encoding='utf-8')
