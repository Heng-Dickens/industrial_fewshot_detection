# YOLOE → Qwen3.8-27B 配对实验

数据集：chache_val100。YOLOE原始10shot视觉提示，候选下限0.001，seed=0；非新微调权重。

用户指定候选阈值：0.01；全图上限：500。开发集仅核查召回变化，不再搜索阈值。
大模型参考：每类1张固定支持图裁图；temperature=0.1，整图+编号框+上下文裁图，原分数不展示。
B1只筛选；B1D在B1基础上按Qwen确认的同类同实例关系去重，保留原分数最高的框；B2只改B1D框坐标。
AP用全部保存分数，P/R/F2及TP/FP/FN为score≥0.5、IoU≥0.5。AP/P/R/F2以百分数显示。

| 阶段 | 框数 | AP50 | AP75 | AP50:95 | AP85 | AP90 | AP95 | 四项≤20% AP | P | R | F2 | TP/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| source | 41060 | 35.34 | 19.20 | 20.96 | 10.98 | 3.33 | 0.75 | 25.31 | 26.67 | 42.11 | 37.74 | 32/88/44 |
| baseline | 13161 | 35.33 | 19.20 | 20.96 | 10.98 | 3.33 | 0.75 | 25.31 | 26.67 | 42.11 | 37.74 | 32/88/44 |
| filtered | 533 | 57.54 | 27.90 | 31.67 | 15.88 | 4.33 | 1.12 | 35.57 | 78.95 | 39.47 | 43.86 | 30/8/46 |
| deduplicated | 515 | 57.56 | 27.90 | 31.68 | 15.88 | 4.33 | 1.12 | 35.58 | 78.95 | 39.47 | 43.86 | 30/8/46 |
| refined | 515 | 56.99 | 33.14 | 32.33 | 14.06 | 2.88 | 0.45 | 35.61 | 78.95 | 39.47 | 43.86 | 30/8/46 |

与B0比较（百分点）：

- filtered：AP50 +22.22，AP50:95 +10.72，召回 -2.63；FP净减少 80。
- deduplicated：AP50 +22.24，AP50:95 +10.72，召回 -2.63；FP净减少 80。

B2−B1D：AP50:95 +0.65，AP85 -1.82 个百分点。

## 审核审计

```json
{
  "selection": {
    "threshold": 0.01,
    "limit": 500,
    "candidates": 13161,
    "recall50": 0.881578947368421,
    "recall85": 0.2894736842105263,
    "box20_coverage": 0.6710526315789473,
    "lost_recall50": 0.013157894736842105,
    "selection_policy": "user_fixed_floor",
    "criterion": "User fixed score>=0.01; no additional global truncation beyond source 100/class (500 total). Dev recall loss is reported, not optimized away.",
    "source_sha256": "adc640c20306ba49b9b9a4ed24e8919f67449a8add83cf1a172c42ff4c57cf5d"
  },
  "split_audit": {
    "exact_image_overlap": 0,
    "shared_sequences": [
      "s00_00000",
      "s00_00008",
      "s01_00000",
      "s02_00002",
      "s02_00006",
      "s03_00000",
      "s04_00001",
      "s04_00004"
    ],
    "limitation": "Existing cohorts may share video sequences; no claim of sequence-independent generalization."
  },
  "filter_accounting": {
    "requests": 4,
    "cached": 826,
    "recovered_cached": 0,
    "failed": 0,
    "retries": 0,
    "input_tokens": 24390,
    "output_tokens": 2816,
    "seconds": 76.61185838701203,
    "wall_seconds": 24.37241555703804
  },
  "decisions": {
    "remove": 12628,
    "uncertain": 249,
    "keep": 284
  },
  "duplicate_edges": 18,
  "invariant_check": "passed: retained coordinates/classes/scores/order unchanged",
  "refine_accounting": {
    "requests": 3,
    "cached": 76,
    "recovered_cached": 0,
    "failed": 0,
    "retries": 0,
    "input_tokens": 11763,
    "output_tokens": 801,
    "seconds": 20.449520613066852,
    "wall_seconds": 8.052804304752499
  },
  "refined_boxes": 213
}
```

限制：固定测试集已有历史结果；开发/测试可能共享视频序列，详见split_audit.json，不宣称序列独立泛化。
已保留不确定与请求失败的原框；失败率见审计，存在失败时不能称为完整成功的Qwen审核。
各阶段metrics.json含完整逐类和各置信度指标；原始请求与回答位于outputs/_shared/runtime/ovd_mllm_filtering_20261030。
