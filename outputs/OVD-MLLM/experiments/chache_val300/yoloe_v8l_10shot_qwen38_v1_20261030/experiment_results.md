# YOLOE → Qwen3.8-27B 配对实验

数据集：chache_val300。YOLOE原始10shot视觉提示，候选下限0.001，seed=0；非新微调权重。

用户指定候选阈值：0.01；全图上限：500。开发集仅核查召回变化，不再搜索阈值。
大模型参考：每类1张固定支持图裁图；temperature=0.1，整图+编号框+上下文裁图，原分数不展示。
B1只筛选；B1D在B1基础上按Qwen确认的同类同实例关系去重，保留原分数最高的框；B2只改B1D框坐标。
AP用全部保存分数，P/R/F2及TP/FP/FN为score≥0.5、IoU≥0.5。AP/P/R/F2以百分数显示。

| 阶段 | 框数 | AP50 | AP75 | AP50:95 | AP85 | AP90 | AP95 | 四项≤20% AP | P | R | F2 | TP/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| source | 122211 | 29.00 | 17.54 | 18.04 | 10.94 | 4.70 | 1.15 | 17.89 | 22.26 | 37.77 | 33.15 | 71/248/117 |
| baseline | 38252 | 28.98 | 17.53 | 18.03 | 10.94 | 4.70 | 1.15 | 17.88 | 22.26 | 37.77 | 33.15 | 71/248/117 |
| filtered | 1812 | 49.31 | 27.16 | 27.92 | 16.02 | 7.34 | 0.85 | 27.96 | 80.00 | 31.91 | 36.28 | 60/15/128 |
| deduplicated | 1766 | 49.38 | 27.17 | 27.93 | 16.02 | 7.34 | 0.85 | 27.96 | 80.00 | 31.91 | 36.28 | 60/15/128 |
| refined | 1766 | 52.20 | 25.19 | 27.86 | 14.36 | 8.51 | 1.09 | 25.60 | 80.00 | 31.91 | 36.28 | 60/15/128 |

与B0比较（百分点）：

- filtered：AP50 +20.33，AP50:95 +9.88，召回 -5.85；FP净减少 233。
- deduplicated：AP50 +20.40，AP50:95 +9.90，召回 -5.85；FP净减少 233。

B2−B1D：AP50:95 -0.07，AP85 -1.66 个百分点。

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
    "cached": 2424,
    "recovered_cached": 0,
    "failed": 0,
    "retries": 1,
    "input_tokens": 14906,
    "output_tokens": 2771,
    "seconds": 81.80779114551842,
    "wall_seconds": 46.02624432975426
  },
  "decisions": {
    "remove": 36440,
    "uncertain": 1100,
    "keep": 712
  },
  "duplicate_edges": 46,
  "invariant_check": "passed: retained coordinates/classes/scores/order unchanged",
  "refine_accounting": {
    "requests": 3,
    "cached": 225,
    "recovered_cached": 0,
    "failed": 0,
    "retries": 0,
    "input_tokens": 15660,
    "output_tokens": 2099,
    "seconds": 41.78111177450046,
    "wall_seconds": 22.653855229262263
  },
  "refined_boxes": 590
}
```

限制：固定测试集已有历史结果；开发/测试可能共享视频序列，详见split_audit.json，不宣称序列独立泛化。
已保留不确定与请求失败的原框；失败率见审计，存在失败时不能称为完整成功的Qwen审核。
各阶段metrics.json含完整逐类和各置信度指标；原始请求与回答位于outputs/_shared/runtime/ovd_mllm_filtering_20261030。
