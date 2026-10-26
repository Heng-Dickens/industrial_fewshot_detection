# 按测试数据集与提示词查找结果

项目 `outputs/runs` 使用以下结构：

```text
outputs/runs/
  prompt_results.md                 # 全部数据集的提示词与指标索引
  diaowu_99/                       # 原来的 99 张吊物测试集
    dataset.json                   # 真实图像集合、真值及数据指纹
    MLLM提示_v1/                   # 原来的严格可见提示词
      prompt_version.json          # 完整文字、复核模板、指纹、版本说明
      qwen3.8-27b_5shot/
        comparison.txt
        原图+裁剪图_无负例/
          prompt.txt
          prompt.json
          evaluation.txt
          metrics.json
          records.jsonl
          predictions/
          predictions_score_0.7/
          feedback/
  diaowu_val300/                   # 新的 300 张吊物验证集
    视觉提示/                       # YOLOE 参考图，不使用文字规则
      yoloe-11l-seg_5shot/          # 直接保存本次结果
    监督检测/
      jsty_hanging_object_det1_0_0_0_1_fp32_监督检测/
    文字提示_v1/                   # GroundingDINO / YOLOE 文字或联合提示
      grounding-dino-base_0shot/
    MLLM提示_v1/                   # 保留旧提示词的基线批次
      qwen3.8-27b_5shot/
    MLLM提示_v2/                   # 执行新版提示词后创建
      qwen3.8-27b_5shot/
```

普通模型的文件直接放在“模型与 shot”目录下。Qwen 对比实验多一层有明确含义的参考输入组合目录，保留每组独立结果。相同条件重跑仍用“第2次”等编号保留历史。

数据集目录按实际参与推理与评估的图像、图像内容和真值区分。更换测试集时会创建独立目录；同名但内容不同的数据集使用“第2套”等名称，具体内容见 `dataset.json`。少量 `--limit` 调试与完整测试也会隔离。

文字及复核模板相同才属于同一个提示词版本。修改任何实际文字内容后，自动选择下一个 `vN` 目录，缓存也按真实输入指纹重新判断，不能复用旧文字产生的预测。版本号在各数据集内独立编号，具体语义以 `prompt_version.json` 为准。纯视觉和监督检测分别归档，不标成新版文字实验。

本次吊物 MLLM v2 允许少量遮挡：吊索、吊绳、吊具与吊物主体及连接关系仍可辨认时继续标注；按实例判断，不因局部人员遮挡而清空整图，也不为避开人物而截断组件框。严重遮挡、连接不明或边界无法可靠确定时仍不猜测。空钩、吊物形状等其他规则保持原定义。

当前已开始的 val300 基线批次使用冻结的旧提示词继续完成。新版使用：

```powershell
python main.py --config configs/specific/mllm_val300_partial_occlusion.yaml
```

该配置明确为 Qwen3.8-27B、5-shot、15 组、300 张图，继续使用现有 CC Switch 转发，并同时给出 0.5 / 0.7 的结果。它与旧版结果分目录保存。修改提示词及整理历史目录本身不会触发新版远程推理。

历史未完成分组的原始文件存入 `outputs/archive/interrupted_runs`，完成基线时从成功缓存恢复，活动目录不产生重复残缺结果。更早的遗留数据在 `outputs/archive/runs_legacy`，运行辅助文件在 `outputs/runtime`。路径迁移表为 `outputs/result_layout_migration.json`，修改前的元数据备份在 `.planning/result_layout_backup`。历史提示词和指标原字节保留，不把新规则归因给旧指标。

自定义项目外输出目录默认保留原布局；可配置 `results_layout: true` 使用相同分组规则。项目内 `outputs/runs` 默认启用，可显式设为 `false` 用于独立兼容脚本。
