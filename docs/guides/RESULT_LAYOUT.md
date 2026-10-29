# 按测试数据集与提示词查找结果

全部实验入口为 [outputs/README.md](../../outputs/README.md)。每个 `outputs/runs/<数据集>/` 使用统一的 `results_summary.md`、`results_summary.csv`、`results_summary.json`，覆盖该数据集的全部已有算法，不限于 MLLM。

```bash
python scripts/summarize_results.py
python scripts/summarize_results.py --dataset chache_val300
```

脚本只读取保存的运行和指标，不调用模型或重算 AP。合并五类的实验占一行，单类结果作为来源和每类 AP50 列保留；不同数据指纹、类别范围和未完成状态明确标注。数据、逐图记录在本机存在时检查图像哈希及各目标的记录覆盖；仅同步了指标时注明未进行逐图核验。CSV采用UTF-8 BOM和单个LF换行，率指标的 `_pct` 列使用百分数，JSON保留0～1原始值。

原有 `summarize_chache_results.py`、`summarize_diaowu_results.py` 命令也更新这套统一总表。总表更新前的原文件按内容哈希保存到 `outputs/archive/summary_snapshots/`。各实验原目录和专题分析路径保持稳定；缓存、环境检查、中断归档不计入正式实验数。

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

五类特种车辆 WeDetect 对比按“数据集 → 算法 → 实验细节”归档。同一算法的不同提示词放在同一个实验目录中：

```text
outputs/runs/chache_val300/
  dataset.json
  prompt_results.md
  WeDetect/                        # 统一算法目录
    method.json
    experiment_results.md          # 每组一行的五类合并指标
    prompt_results.md              # 所有类别的提示词及结果链接
    wedetect_base_0shot_prompt_v1/  # 简短类别词
      prompt_version.json
      metrics.json                 # 五类合并指标
      records.jsonl                # 五类共1500条记录，对应同一300张图片
      prompt.txt                   # 五个类别的实际文字
      experiment.json              # 分类配置与对比报告路径
      Chache1/                     # 单类运行的提示词、预测、指标和反馈
      Chache2/
      Chache3/
      Chache4/
      Qianyinche/
    wedetect_base_0shot_prompt_v2/  # 具体类别名
    wedetect_base_0shot_prompt_v3/  # 短描述
    wedetect_base_0shot_prompt_v4/  # 长描述
```

上述 v1～v4 对应 2026-10-27 的这次对比；以后以实际保存的 `prompt_version.json` 为准。
`WeDetect/.prompts/` 保存按实际文字指纹分配的提示词版本。修改提示词会在同一算法目录内生成新版本，
普通 WeDetect 运行和对比脚本均使用此规则；不同权重和重复运行仍由具体实验目录名区分。
`outputs/comparisons/wedetect_chache_prompts_20261027/` 只保存对比报告、指标表、图表、
画框对照图和运行目录索引。配置、派生清单、缓存与日志位于
`outputs/runtime/wedetect_chache_prompts_20261027/`；中断残留归入 `outputs/archive/interrupted_runs/`。
迁移记录保存在 runtime 下的 `result_layout_migration.json` 和 `method_layout_migration.json`，
迁移已完成，核验后的一次性 `.planning` 备份已清理，迁移记录继续保留。原来分散在 `文字提示_v1～v4` 的目录索引存入
`outputs/archive/previous_prompt_layout/`；实际预测仅保留在新的 WeDetect 实验目录。

`scripts/run_wedetect_chache_prompts.py` 后续直接写入该标准目录，`--output` 仅指定对比报告位置。
已有完整结果会复用，使用 `--phase status` 查看状态；为新提示词对比选择新的 `--output`，避免覆盖历史对比报告。

PET-DINO 视觉提示也按算法归组，diaowu 与 chache 分别保存：

```text
outputs/runs/<diaowu_val300 或 chache_val300>/PET-DINO/
  experiment_results.md
  pet_dino_swin_t_1shot/
  pet_dino_swin_t_5shot/
  pet_dino_swin_t_10shot/
```

shot 为每类独立参考图片数，固定 seed=0，使用冻结且逐级包含的支持集。
chache 的每个 shot 目录包含五类合并结果，以及 Chache1～Chache4、Qianyinche 单类子目录；
每类都推理完整 300 张图，包括负例，不按待测图真值选择类别。
`scripts/run_pet_dino_shots.py --phase prepare` 检查两套数据及 18 项单类任务，
`--phase complete` 使用本机 GPU 调度完整推理和评估，`--phase status` 查看进度。
配置、只调整路径的支持清单副本、缓存和日志位于
`outputs/runtime/pet_dino_shots_20261027/`，跨场景指标表位于同名 `outputs/comparisons/` 目录。
同一计划下重跑会复用已完成任务；计划不一致时停止，避免混入已有实验。

与上述结果对齐的 YOLOE / GroundingDINO 重跑使用：

```text
outputs/runs/<dataset>/YOLOE/yoloe-v8l-seg_{1,5,10}shot/
outputs/runs/<dataset>/GroundingDINO/grounding-dino-base_0shot_prompt_vN/
```

YOLOE 逐项复用 PET-DINO 的参考图、框和 seed；GroundingDINO 逐字复用对应 WeDetect
运行快照中的中文提示词（diaowu 两组、chache 四组）。全部使用冻结的完整300张验证图，
chache 每类独立推理后合并。候选下限0.001、NMS=0.7、每类最多100框、评估IoU=0.5和
P/R阈值0.5对齐。模型各自的图像预处理与特征聚合保留：YOLOE为640方形输入和
normalized_mean，GroundingDINO为本地800/1333预处理及text_threshold=0.25。
这些模型内部参数不宣称相同。

`scripts/run_aligned_baselines.py --phase prepare` 校验输入对应关系；
`--phase complete` 运行12组实验、40项单类任务，共12000次图像推理。
本轮显式禁用旧预测缓存；同一计划中已完成任务可跳过以便续跑。
配置、进度和日志位于 `outputs/runtime/aligned_baselines_20261027/`，
对齐比较表位于 `outputs/comparisons/aligned_baselines_20261027/report.md`。
每个算法目录的 `experiment_results.md` 也包含对应 PET-DINO 或 WeDetect 的基线指标。

数据集目录按实际参与推理与评估的图像、图像内容和真值区分。更换测试集时会创建独立目录；同名但内容不同的数据集使用“第2套”等名称，具体内容见 `dataset.json`。少量 `--limit` 调试与完整测试也会隔离。

文字及复核模板相同才属于同一个提示词版本。修改任何实际文字内容后，自动选择下一个 `vN` 目录，缓存也按真实输入指纹重新判断，不能复用旧文字产生的预测。版本号在各数据集内独立编号，具体语义以 `prompt_version.json` 为准。纯视觉和监督检测分别归档，不标成新版文字实验。

本次吊物 MLLM v2 允许少量遮挡：吊索、吊绳、吊具与吊物主体及连接关系仍可辨认时继续标注；按实例判断，不因局部人员遮挡而清空整图，也不为避开人物而截断组件框。严重遮挡、连接不明或边界无法可靠确定时仍不猜测。空钩、吊物形状等其他规则保持原定义。

当前已开始的 val300 基线批次使用冻结的旧提示词继续完成。新版使用：

```powershell
python main.py --config configs/specific/mllm_val300_partial_occlusion.yaml
```

该配置明确为 Qwen3.8-27B、5-shot、15 组、300 张图，继续使用现有 CC Switch 转发，并同时给出 0.5 / 0.7 的结果。它与旧版结果分目录保存。修改提示词及整理历史目录本身不会触发新版远程推理。

历史未完成分组的原始文件存入 `outputs/archive/interrupted_runs`，完成基线时从成功缓存恢复，活动目录不产生重复残缺结果。更早的遗留数据在 `outputs/archive/runs_legacy`，运行辅助文件在 `outputs/runtime`。一次性迁移备份在完成核验后可清理，历史提示词和指标原字节保留，不把新规则归因给旧指标。

框误差分析统一保留 [总体表](../../outputs/comparisons/iou85_box20_20261028/README.md) 及其 CSV/JSON；其中已含各方法最佳配置，不再保留一份独立的最佳实验结果。旧的修框上界分析已移除，当前“四项误差≤20%”使用原预测框。

已完成的本地检测缓存可在逐条验证与正式 `records.jsonl` 一致后清理；缓存只是加速副本。保留正式记录、`run.json`、配置、支持集、指标、预测文件和 MLLM 成功缓存，以便重新评估、复用或追溯实验。

自定义项目外输出目录默认保留原布局；可配置 `results_layout: true` 使用相同分组规则。项目内 `outputs/runs` 默认启用，可显式设为 `false` 用于独立兼容脚本。
