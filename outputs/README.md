# 实验结果导航

[全局代表方案与最终结论](FINAL_COMPARISON.md) · [按子任务比较](_shared/comparisons/final_results/README.md)

按方法查看实验；每个方法的结果、比较文档和复现脚本放在同一目录。微调位于该方法的 finetune/。

| 方法 | 实验结果与比较 | 微调 |
|---|---|---|
| GroundingDINO | [GroundingDINO](GroundingDINO/README.md) | — |
| JSTY-YOLO | [JSTY-YOLO](JSTY-YOLO/README.md) | — |
| OVD-MLLM | [OVD-MLLM](OVD-MLLM/README.md) | — |
| PET-DINO | [PET-DINO](PET-DINO/README.md) | — |
| Qwen | [Qwen](Qwen/README.md) | — |
| WeDetect | [WeDetect](WeDetect/README.md) | — |
| YOLOE | [YOLOE](YOLOE/README.md) | [微调](YOLOE/finetune/README.md) |

## 按队列的原始索引

| 数据集 | 全方法结果 |
|---|---|
| chache_val100 | [总表](_shared/datasets/chache_val100/results_summary.md) |
| chache_val300 | [总表](_shared/datasets/chache_val300/results_summary.md) |
| diaowu_99 | [总表](_shared/datasets/diaowu_99/results_summary.md) |
| diaowu_val100 | [总表](_shared/datasets/diaowu_val100/results_summary.md) |
| diaowu_val300 | [总表](_shared/datasets/diaowu_val300/results_summary.md) |

[全部运行明细（完整保留）](ALL_RESULTS.md) · [完整CSV指标](ALL_RESULTS.csv)

[YOLOv8与YOLOE：两场景5shot训练及测试推理对照](YOLOE/finetune/5SHOT_COMPARISON.md)

```text
outputs/
  <方法>/
    README.md
    experiments/<数据集>/<实验>/
    comparisons/<数据集或专题>/
    scripts/
    finetune/<数据集>/runs/与comparisons/
  _shared/  # 跨方法数据身份、总表、缓存、日志、环境与历史归档
```

原始指标、逐图预测、权重及请求回答保留；历史路径通过迁移清单解析。
更新导航与各方法总表：`python scripts/summarize_results.py`。
