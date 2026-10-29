# 实验结果导航

按数据集打开总表。各算法、shot、提示词、参考输入组合和重复运行均在同一张表中；CSV适合筛选与跨机器比较。

| 数据集 | 图片数 | 实验数 | 算法 | 总表 | CSV |
|---|---:|---:|---|---|---|
| chache_val300 | 300 | 36 | GroundingDINO、MLLM、PET-DINO、WeDetect、YOLOE | [打开](runs/chache_val300/results_summary.md) | [下载](runs/chache_val300/results_summary.csv) |
| diaowu_99 | 99 | 45 | MLLM | [打开](runs/diaowu_99/results_summary.md) | [下载](runs/diaowu_99/results_summary.csv) |
| diaowu_val300 | 300 | 33 | GroundingDINO、MLLM、PET-DINO、WeDetect、YOLOE | [打开](runs/diaowu_val300/results_summary.md) | [下载](runs/diaowu_val300/results_summary.csv) |

框误差专题统一查看 [AP85/90/95 与四项误差≤20%总体表](comparisons/iou85_box20_20261028/README.md)，其中包含全部实验及各方法最佳配置，无需另查最佳实验副本。

```text
outputs/
  README.md                         # 当前导航
  runs/<数据集>/
    dataset.json                    # 数据版本身份
    results_summary.md/csv/json      # 该数据集全部实验的统一总表
    <算法或提示词版本>/<实验>/        # 原始运行、指标、预测与prompt
  comparisons/<专题>/               # 跨算法/shot/提示词的专题分析及对比图
  runtime/                          # 运行配置、清单副本、缓存和日志
  environment_setup/                # 环境验证日志
  archive/                          # 中断、旧布局及总表覆盖前的原文件快照
```

原始运行路径保持稳定，专题分析继续保留；总表通过相对链接组织它们。历史总表覆盖前的原字节保存在 `archive/summary_snapshots/<数据集>/<SHA256>/`。不把旧总表或单类子运行当成新实验重复统计。

目录迁移完成后的一次性备份和已被正式记录覆盖的本地检测缓存已清理。`runtime` 保留运行配置、支持集、迁移记录和验证报告；正式逐图预测及 MLLM 成功缓存保留。

更新全部数据集：`python scripts/summarize_results.py`。
仅更新一个数据集：`python scripts/summarize_results.py --dataset chache_val300`。
即使其他机器只有已同步的报告和元数据，也能生成表格；缺少逐图记录时会明确标注核验限制。
