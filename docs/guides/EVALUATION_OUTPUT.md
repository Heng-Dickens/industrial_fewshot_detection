# 检测评估报告

项目结果目录现按“测试数据集 / 提示词版本 / 模型与 shot”组织，完整规则和旧、新遮挡提示词对比入口见 [结果目录说明](RESULT_LAYOUT.md)。普通单组目录直接存文件，Qwen 对比实验再按参考输入组合分组。

正式运行与 `scripts/evaluate.py RUN_DIR` 默认输出：官方 COCO 的 12 行 AP／AR 摘要、按类别的 AP／Recall／F2 表、VOC 风格的 TP／检测数／真值数／precision／recall／F2／error 表。每个实验组分别显示结果。只读 `--dry-run` 默认继续输出 JSON，不会把规划数据显示为评估成绩。

```powershell
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/mllm_fewshot.yaml
# 推理已完成时，只补算评估，不请求模型；RUN_DIR 为实际单次运行目录
.\.venv-yoloe\Scripts\python.exe scripts/evaluate.py RUN_DIR
# 程序读取完整 JSON 时显式指定
.\.venv-yoloe\Scripts\python.exe scripts/evaluate.py RUN_DIR --json
```

单次运行保存 `evaluation.txt`；参考实验汇总保存 `comparison.txt`；模型比较保存 `detector_comparison.txt`。原有 JSON／CSV 输出继续保留。`--report-format text`／`json` 控制主入口的终端格式。

`evaluation.score_threshold` 控制默认最终框输出；`evaluation.score_thresholds` 指定同时提供的其他阈值，默认 `[0.5, 0.7]`。两档共用一次模型推理：

```yaml
evaluation:
  iou_threshold: 0.5
  score_threshold: 0.5
  score_thresholds: [0.5, 0.7]
  max_detections_per_image: 100
```

默认 0.5 的标注仍直接保存在 `predictions/*.json`；0.7 的标注放在同级 `predictions_score_0.7/*.json`，均只保留 `confidence >= 阈值` 的框。主阈值另有设置时，仍保留它，额外目录按 `predictions_score_阈值` 命名。`evaluation.txt` 和终端分别打印各档 VOC-style 表，`metrics.json.score_threshold_results["0.5"]` / `["0.7"]` 保存各类别及总体 TP、FP、FN、Precision、Recall、F2。标准 COCO AP/AR 使用完整候选框，仅计算和展示一份。

`main.py visualize` 默认按运行保存的主阈值绘制红色预测框，也可以显式选择 0.7；绿色真值框不筛选。可视化 `manifest.json` 记录阈值及每图候选数、显示数、过滤数。调整阈值后用原配置命令重跑，会复用完整预测重新导出各档标注；不要加 `--force`。原始 `records.jsonl`、反馈、缓存仍保留完整候选框。

```powershell
# 为已有实验补充两档结果和额外标注，不调用模型、不改原始记录和默认标注
.\.venv-yoloe\Scripts\python.exe scripts/add_score_threshold_results.py --run-dir RUN_DIR
# 为全部现有实验补充（仅扫描 runs 的模型目录及实验组，不扫描 legacy/runtime）
.\.venv-yoloe\Scripts\python.exe scripts/add_score_threshold_results.py --all-runs
# 分别查看两档框，输出目录须在原始图片、真值和运行目录以外
.\.venv-yoloe\Scripts\python.exe main.py visualize --run-dir RUN_DIR --output-dir outputs/visualizations/score_0.5 --score-threshold 0.5
.\.venv-yoloe\Scripts\python.exe main.py visualize --run-dir RUN_DIR --output-dir outputs/visualizations/score_0.7 --score-threshold 0.7
```

仅执行 `--evaluate-only` 会更新含各档指标的评估报告，不重新导出标注或修改运行时保存的主阈值；需要补出 0.7 标注时使用上述补充脚本。

COCO 摘要由 `pycocotools.COCOeval` 实际计算，IoU 为 0.50:0.05:0.95，maxDets 为 1／10／100。面积使用原图像素坐标中的框面积，small／medium／large 使用 COCO 默认的 32²／96² 分界；没有该档真值时显示 `-1.000`。输入为 LabelMe 矩形框，没有 crowd／ignore 标记。

分类表中的 AP 是 AP50，Recall 是 IoU=0.50、maxDets=100 下的召回率，F2 从同一匹配结果的真实 precision／recall 计算：`5PR/(4P+R)`，不将 AP 当作 precision。AP／Recall／F2 显示百分数。

VOC 风格表沿用配置中的 `evaluation.iou_threshold`、`score_threshold` 和 `max_detections_per_image`，采用连续像素框的一对一匹配；它展示检测计数，不是额外的 VOC2007 十一点 AP。`dets=TP+FP`、`gts=TP+FN`，`error=FP/dets`（0～1），无检测时为 0。mean 行的计数求和、比率按类别宏平均。两张表的筛选口径不同，因此数值可以不同。

新增的官方 COCO 指标保存在 `metrics.json` 的 `coco` 字段，包含 `stats`、`per_class` 和 `summary`；原有 `mAP50` 等兼容指标仍保留，其实现及历史比较口径不变。F2 同时写入原有分类及整体指标。失败请求仍计入原有整图错误分母，报告末尾显示 incomplete 状态和失败数。

若运行环境缺少依赖，在同一个 Python 环境执行 `python -m pip install -r requirements.txt`。评估依赖会在正式推理前检查。
