# OVD 最佳实验：AP85 与框几何误差 20%

生成时间：2026-10-28 13:03:33（北京时间）。读取 8 组独立实验，保留各数据集和原状态。

**口径**

- AP85/90/95：同类别预测按置信度排序，一对一匹配，101 点插值 AP；多类取有 GT 类别的宏平均。直接读取原指标中的 `ap_by_iou`，不先按 IoU 筛掉误检。
- AP85:95 = (AP85 + AP90 + AP95) / 3。AP 使用全部已保存候选分数；置信度阈值统计是另外的工作点，不是 AP。候选在模型输出前被删掉的部分无法恢复。
- 宽误差 = |预测宽 − GT宽| / GT宽；高误差同理。横向中心偏移 = |预测中心x − GT中心x| / GT宽；纵向偏移以 GT高归一化。四项均 ≤ 20% 才合格；未额外要求 IoU ≥ 0.85。
- 同类预测按置信度降序，一张图内每个 GT 最多使用一次；从满足四项误差约束且未匹配的 GT 中选最大 IoU。误检、重复框、错分类不重复计数。
- 数量表为“合格框/选中预测框数（预测合格率）；GT覆盖率”。漏检保留在 GT 分母。无预测时合格数为 0、预测合格率未定义，GT覆盖率为 0。每图每类沿用保存的 maxDets 上限。
- `confidence_thresholds.csv` 给出 score≥0.0/0.1/…/1.0 的累计统计；`confidence_bins.csv` 给出独立分数区间（左闭右开，最后区间含1.0）。分箱各自重新一对一匹配，不可将分箱合格数相加当成累计结果。
- CSV 中带 `_pct` 的字段为百分数，其余率字段和 JSON 使用 0–1。缺原始坐标显示“—”，不能从 AP、TP/FP/FN 或平均 IoU 反推误差合格数。
- `size20_on_iou50_count` / `offset20_on_iou50_count` 是 IoU≥0.5 匹配框中分别满足宽高/中心偏移约束的辅助数量；`joint20_count` 独立按四项联合约束匹配。

[AP及主统计 CSV](ap_summary.csv) · [置信度累计 CSV](confidence_thresholds.csv) · [置信度分箱 CSV](confidence_bins.csv) · [各类别AP CSV](class_ap.csv) · [完整汇总JSON](analysis.json)

仅比较 GroundingDINO、PET-DINO、WeDetect、YOLOE；每个数据集、每种方法选取允许全数据集比较的完整实验中 AP85 最高的一组。AP85 并列时比较 AP85:95，再并列保持原总表顺序。这里的最佳配置按 AP85 选择，可能与 AP50 最佳配置不同。

## chache_val300

共 4 组；保存 AP85 的 4 组；完成框误差统计的 0 组。

| # | 方法/模型 | prompt / shot / 输入 / 负例 | AP85% | AP90% | AP95% | AP85:95% | score≥0.5 四项≤20% | score≥0.7 四项≤20% | 状态 |
|---:|---|---|---:|---:|---:|---:|---|---|---|
| chache_val300-01 | GroundingDINO / grounding-dino-base | [prompt_v1 / 0shot / text_only / 0负](../../runs/chache_val300/GroundingDINO/grounding-dino-base_0shot_prompt_v1/metrics.json) | 7.34 | 6.78 | 1.11 | 5.08 | — | — | 完成；缺坐标 |
| chache_val300-27 | PET-DINO / pet_dino_swin_t | [视觉提示 / 1shot / 视觉参考 / 0负](../../runs/chache_val300/PET-DINO/pet_dino_swin_t_1shot/metrics.json) | 13.46 | 12.27 | 3.91 | 9.88 | — | — | 完成；缺坐标 |
| chache_val300-33 | WeDetect / wedetect_base | [prompt_v4 / 0shot / text_only / 0负](../../runs/chache_val300/WeDetect/wedetect_base_0shot_prompt_v4/metrics.json) | 25.83 | 20.00 | 3.94 | 16.59 | — | — | 完成；缺坐标 |
| chache_val300-36 | YOLOE / yoloe-v8l-seg | [视觉提示 / 10shot / 视觉参考 / 0负](../../runs/chache_val300/YOLOE/yoloe-v8l-seg_10shot/metrics.json) | 10.94 | 4.70 | 1.15 | 5.60 | — | — | 完成；缺坐标 |

**各方法在本数据集的最高 AP85（仅原总表允许比较的完整实验）**

| 方法 | AP85% | AP90% | AP95% | shot / 输入 / 负例 | 实验ID |
|---|---:|---:|---:|---|---|
| GroundingDINO | 7.34 | 6.78 | 1.11 | 0 / text_only / 0 | chache_val300-01 |
| PET-DINO | 13.46 | 12.27 | 3.91 | 1 / 视觉参考 / 0 | chache_val300-27 |
| WeDetect | 25.83 | 20.00 | 3.94 | 0 / text_only / 0 | chache_val300-33 |
| YOLOE | 10.94 | 4.70 | 1.15 | 10 / 视觉参考 / 0 | chache_val300-36 |

**无法计算框误差的实验**

- chache_val300-01 `GroundingDINO/grounding-dino-base_0shot_prompt_v1`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-27 `PET-DINO/pet_dino_swin_t_1shot`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-33 `WeDetect/wedetect_base_0shot_prompt_v4`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-36 `YOLOE/yoloe-v8l-seg_10shot`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。

## diaowu_val300

共 4 组；保存 AP85 的 4 组；完成框误差统计的 0 组。

| # | 方法/模型 | prompt / shot / 输入 / 负例 | AP85% | AP90% | AP95% | AP85:95% | score≥0.5 四项≤20% | score≥0.7 四项≤20% | 状态 |
|---:|---|---|---:|---:|---:|---:|---|---|---|
| diaowu_val300-02 | GroundingDINO / grounding-dino-base | [prompt_v2 / 0shot / text_only / 0负](../../runs/diaowu_val300/GroundingDINO/grounding-dino-base_0shot_prompt_v2/metrics.json) | 0.08 | 0.05 | 0.00 | 0.05 | — | — | 完成；缺坐标 |
| diaowu_val300-26 | PET-DINO / pet_dino_swin_t | [视觉提示 / 1shot / 视觉参考 / 0负](../../runs/diaowu_val300/PET-DINO/pet_dino_swin_t_1shot/metrics.json) | 8.68 | 5.25 | 1.33 | 5.09 | — | — | 完成；缺坐标 |
| diaowu_val300-29 | WeDetect / wedetect_base | [文字提示_v1 / 0shot / text_only / 0负](../../runs/diaowu_val300/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/wedetect_base_0shot/metrics.json) | 0.99 | 0.01 | 0.00 | 0.33 | — | — | 完成；缺坐标 |
| diaowu_val300-31 | YOLOE / yoloe-v8l-seg | [视觉提示 / 1shot / 视觉参考 / 0负](../../runs/diaowu_val300/YOLOE/yoloe-v8l-seg_1shot/metrics.json) | 15.56 | 6.14 | 0.63 | 7.44 | — | — | 完成；缺坐标 |

**各方法在本数据集的最高 AP85（仅原总表允许比较的完整实验）**

| 方法 | AP85% | AP90% | AP95% | shot / 输入 / 负例 | 实验ID |
|---|---:|---:|---:|---|---|
| GroundingDINO | 0.08 | 0.05 | 0.00 | 0 / text_only / 0 | diaowu_val300-02 |
| PET-DINO | 8.68 | 5.25 | 1.33 | 1 / 视觉参考 / 0 | diaowu_val300-26 |
| WeDetect | 0.99 | 0.01 | 0.00 | 0 / text_only / 0 | diaowu_val300-29 |
| YOLOE | 15.56 | 6.14 | 0.63 | 1 / 视觉参考 / 0 | diaowu_val300-31 |

**无法计算框误差的实验**

- diaowu_val300-02 `GroundingDINO/grounding-dino-base_0shot_prompt_v2`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-26 `PET-DINO/pet_dino_swin_t_1shot`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-29 `文字提示_v1/wedetect_base_0shot`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-31 `YOLOE/yoloe-v8l-seg_1shot`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。

## 复算与数据来源

`python scripts/analyze_box_precision.py --ovd-best --dataset chache_val300 --dataset diaowu_val300`；只分析一个数据集可使用 `--dataset diaowu_val300`。

原始预测、GT、metrics 和已有 results_summary 均保持原字节。数据来源和 SHA256 记录在 `analysis.json`。框误差复算要求 GT 清单与运行时数据指纹相符，并验证逐图记录完整性及图像哈希；没有复核远端机器上尚未同步的原始预测。
