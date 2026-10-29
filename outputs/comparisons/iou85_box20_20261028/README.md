# AP85 与框几何误差 20% 统计

生成时间：2026-10-29 09:05:47（北京时间）。读取 114 组独立实验，保留各数据集和原状态。

**口径**

- AP85/90/95：同类别预测按置信度排序，一对一匹配，101 点插值 AP；多类取有 GT 类别的宏平均。直接读取原指标中的 `ap_by_iou`，不先按 IoU 筛掉误检。
- AP85:95 = (AP85 + AP90 + AP95) / 3。AP 使用全部已保存候选分数；置信度阈值统计是另外的工作点，不是 AP。候选在模型输出前被删掉的部分无法恢复。
- 宽误差 = |预测宽 − GT宽| / GT宽；高误差同理。横向中心偏移 = |预测中心x − GT中心x| / GT宽；纵向偏移以 GT高归一化。四项均 ≤ 20% 才合格；未额外要求 IoU ≥ 0.85。
- **四项≤20% AP**：预测框不移动，以上四项同时满足20%容差即为可匹配框；按原置信度排序、同类一对一匹配，计算101点插值AP，再对有GT的类别取宏平均。该列采用几何容差匹配，不是IoU=0.20的AP，也不是GT辅助修框值或理论上界。
- 同类预测按置信度降序，一张图内每个 GT 最多使用一次；从满足四项误差约束且未匹配的 GT 中选最大 IoU。误检、重复框、错分类不重复计数。
- 数量表为“合格框/选中预测框数（预测合格率）；GT覆盖率”。漏检保留在 GT 分母。无预测时合格数为 0、预测合格率未定义，GT覆盖率为 0。每图每类沿用保存的 maxDets 上限。
- `confidence_thresholds.csv` 给出 score≥0.0/0.1/…/1.0 的累计统计；`confidence_bins.csv` 给出独立分数区间（左闭右开，最后区间含1.0）。分箱各自重新一对一匹配，不可将分箱合格数相加当成累计结果。
- CSV 中带 `_pct` 的字段为百分数，其余率字段和 JSON 使用 0–1。缺原始坐标显示“—”，不能从 AP、TP/FP/FN 或平均 IoU 反推误差合格数。
- 缺坐标的实验保留原表的历史合格数量；部分来源文件哈希与历史报告不同，数量未在本机重新复核，详见JSON中的 `geometry_statistics_source`。新“四项≤20% AP”显示“—”，不由合格率反推。
- `size20_on_iou50_count` / `offset20_on_iou50_count` 是 IoU≥0.5 匹配框中分别满足宽高/中心偏移约束的辅助数量；`joint20_count` 独立按四项联合约束匹配。

[AP及主统计 CSV](ap_summary.csv) · [置信度累计 CSV](confidence_thresholds.csv) · [置信度分箱 CSV](confidence_bins.csv) · [各类别AP CSV](class_ap.csv) · [完整汇总JSON](analysis.json)

## chache_val300

共 36 组；保存 AP85 的 35 组；完成框误差统计的 14 组。

| # | 方法/模型 | prompt / shot / 输入 / 负例 | AP85% | AP90% | AP95% | AP85:95% | 四项≤20% AP% | score≥0.5 四项≤20% | score≥0.7 四项≤20% | 状态 |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| chache_val300-01 | GroundingDINO / grounding-dino-base | [prompt_v1 / 0shot / text_only / 0负](../../runs/chache_val300/GroundingDINO/grounding-dino-base_0shot_prompt_v1/metrics.json) | 7.34 | 6.78 | 1.11 | 5.08 | 9.05 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| chache_val300-02 | GroundingDINO / grounding-dino-base | [prompt_v2 / 0shot / text_only / 0负](../../runs/chache_val300/GroundingDINO/grounding-dino-base_0shot_prompt_v2/metrics.json) | 7.34 | 6.17 | 1.76 | 5.09 | 8.87 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| chache_val300-03 | GroundingDINO / grounding-dino-base | [prompt_v3 / 0shot / text_only / 0负](../../runs/chache_val300/GroundingDINO/grounding-dino-base_0shot_prompt_v3/metrics.json) | 6.23 | 5.08 | 1.91 | 4.40 | 7.45 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| chache_val300-04 | GroundingDINO / grounding-dino-base | [prompt_v4 / 0shot / text_only / 0负](../../runs/chache_val300/GroundingDINO/grounding-dino-base_0shot_prompt_v4/metrics.json) | 2.33 | 1.80 | 0.60 | 1.57 | 2.95 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| chache_val300-05 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 0shot / text_only / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/metrics.json) | 24.90 | 19.21 | 7.50 | 17.20 | — | 79/207（38.16%）；GT 42.02% | 79/206（38.35%）；GT 42.02% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-06 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.24 | 27.26 | 10.01 | 24.84 | — | 106/196（54.08%）；GT 56.38% | 106/196（54.08%）；GT 56.38% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-07 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 0shot / text_only / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/metrics.json) | 19.69 | 9.94 | 5.21 | 11.61 | — | 72/214（33.64%）；GT 38.30% | 72/214（33.64%）；GT 38.30% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-08 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 1shot / O+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 23.91 | 17.59 | 7.20 | 16.23 | — | 91/200（45.50%）；GT 48.40% | 91/200（45.50%）；GT 48.40% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-09 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 1shot / O+C / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 25.27 | 20.42 | 6.84 | 17.51 | — | 87/215（40.47%）；GT 46.28% | 87/215（40.47%）；GT 46.28% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-10 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 1shot / O+C+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 26.77 | 20.14 | 7.33 | 18.08 | — | 91/205（44.39%）；GT 48.40% | 91/205（44.39%）；GT 48.40% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-11 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 1shot / O+C+B+J / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 25.04 | 17.03 | 6.22 | 16.10 | — | 86/194（44.33%）；GT 45.74% | 86/194（44.33%）；GT 45.74% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-12 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 5shot / O+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.05 | 23.96 | 7.68 | 22.90 | — | 109/199（54.77%）；GT 57.98% | 109/199（54.77%）；GT 57.98% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-13 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 5shot / O+B+J / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 31.71 | 24.59 | 7.94 | 21.41 | — | 99/195（50.77%）；GT 52.66% | 99/195（50.77%）；GT 52.66% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-14 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 5shot / O+C / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.58 | 28.60 | 8.92 | 25.03 | — | 110/233（47.21%）；GT 58.51% | 110/233（47.21%）；GT 58.51% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-15 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 5shot / O+C+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 38.46 | 26.67 | 9.35 | 24.83 | — | 115/198（58.08%）；GT 61.17% | 115/198（58.08%）；GT 61.17% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-16 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 5shot / O+C+B+J / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 39.15 | 27.18 | 7.36 | 24.56 | — | 116/205（56.59%）；GT 61.70% | 116/205（56.59%）；GT 61.70% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-17 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 5shot / O+C+J / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 35.98 | 25.39 | 9.23 | 23.53 | — | 118/224（52.68%）；GT 62.77% | 118/224（52.68%）；GT 62.77% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-18 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 5shot / O+J / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 29.41 | 21.62 | 7.20 | 19.41 | — | 97/220（44.09%）；GT 51.60% | 97/220（44.09%）；GT 51.60% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-19 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 10shot / O+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 40.05 | 29.13 | 10.06 | 26.41 | — | 119/197（60.41%）；GT 63.30% | 119/197（60.41%）；GT 63.30% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-20 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 10shot / O+C / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 44.61 | 35.27 | 8.75 | 29.54 | — | 125/213（58.69%）；GT 66.49% | 125/213（58.69%）；GT 66.49% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-21 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 10shot / O+C+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 42.97 | 30.42 | 7.61 | 27.00 | — | 121/196（61.73%）；GT 64.36% | 121/196（61.73%）；GT 64.36% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-22 | MLLM / qwen3.8-27b | [MLLM提示_v2 / 10shot / O+C+B+J / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.86 | 27.00 | 10.40 | 25.09 | — | 116/195（59.49%）；GT 61.70% | 116/195（59.49%）；GT 61.70% | 未完成/失败；缺坐标；数量沿用已存统计 |
| chache_val300-23 | MLLM / qwen3.8-flash | [MLLM提示_v2 / 1shot / O+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/run.json) | — | — | — | — | — | — | — | 待核验；缺坐标 |
| chache_val300-24 | MLLM / qwen3.8-flash | [MLLM提示_v2 / 1shot / O+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_1shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 32.57 | 24.18 | 9.23 | 22.00 | — | 98/239（41.00%）；GT 52.13% | 98/234（41.88%）；GT 52.13% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-25 | MLLM / qwen3.8-flash | [MLLM提示_v2 / 5shot / O+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 36.51 | 26.88 | 7.94 | 23.77 | — | 111/238（46.64%）；GT 59.04% | 111/232（47.84%）；GT 59.04% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-26 | MLLM / qwen3.8-flash | [MLLM提示_v2 / 10shot / O+B / 0负](../../runs/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 38.59 | 27.90 | 9.28 | 25.26 | — | 119/234（50.85%）；GT 63.30% | 119/227（52.42%）；GT 63.30% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| chache_val300-27 | PET-DINO / pet_dino_swin_t | [视觉提示 / 1shot / 视觉参考 / 0负](../../runs/chache_val300/PET-DINO/pet_dino_swin_t_1shot/metrics.json) | 13.46 | 12.27 | 3.91 | 9.88 | 13.70 | 24/62（38.71%）；GT 12.77% | 3/3（100.00%）；GT 1.60% | 完成 |
| chache_val300-28 | PET-DINO / pet_dino_swin_t | [视觉提示 / 5shot / 视觉参考 / 0负](../../runs/chache_val300/PET-DINO/pet_dino_swin_t_5shot/metrics.json) | 12.80 | 8.42 | 3.90 | 8.37 | 13.74 | 2/2（100.00%）；GT 1.06% | 0/0（—%）；GT 0.00% | 完成 |
| chache_val300-29 | PET-DINO / pet_dino_swin_t | [视觉提示 / 10shot / 视觉参考 / 0负](../../runs/chache_val300/PET-DINO/pet_dino_swin_t_10shot/metrics.json) | 11.78 | 10.04 | 6.14 | 9.32 | 14.20 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| chache_val300-30 | WeDetect / wedetect_base | [prompt_v1 / 0shot / text_only / 0负](../../runs/chache_val300/WeDetect/wedetect_base_0shot_prompt_v1/metrics.json) | 9.98 | 6.27 | 3.46 | 6.57 | 13.00 | 31/125（24.80%）；GT 16.49% | 19/60（31.67%）；GT 10.11% | 完成 |
| chache_val300-31 | WeDetect / wedetect_base | [prompt_v2 / 0shot / text_only / 0负](../../runs/chache_val300/WeDetect/wedetect_base_0shot_prompt_v2/metrics.json) | 9.57 | 5.56 | 2.55 | 5.89 | 13.63 | 21/120（17.50%）；GT 11.17% | 14/53（26.42%）；GT 7.45% | 完成 |
| chache_val300-32 | WeDetect / wedetect_base | [prompt_v3 / 0shot / text_only / 0负](../../runs/chache_val300/WeDetect/wedetect_base_0shot_prompt_v3/metrics.json) | 12.62 | 8.46 | 4.20 | 8.43 | 16.01 | 19/61（31.15%）；GT 10.11% | 13/19（68.42%）；GT 6.91% | 完成 |
| chache_val300-33 | WeDetect / wedetect_base | [prompt_v4 / 0shot / text_only / 0负](../../runs/chache_val300/WeDetect/wedetect_base_0shot_prompt_v4/metrics.json) | 25.83 | 20.00 | 3.94 | 16.59 | 28.95 | 29/55（52.73%）；GT 15.43% | 5/10（50.00%）；GT 2.66% | 完成 |
| chache_val300-34 | YOLOE / yoloe-v8l-seg | [视觉提示 / 1shot / 视觉参考 / 0负](../../runs/chache_val300/YOLOE/yoloe-v8l-seg_1shot/metrics.json) | 5.21 | 3.03 | 0.20 | 2.81 | 13.49 | 31/87（35.63%）；GT 16.49% | 16/42（38.10%）；GT 8.51% | 完成 |
| chache_val300-35 | YOLOE / yoloe-v8l-seg | [视觉提示 / 5shot / 视觉参考 / 0负](../../runs/chache_val300/YOLOE/yoloe-v8l-seg_5shot/metrics.json) | 8.21 | 2.97 | 0.38 | 3.85 | 16.47 | 52/264（19.70%）；GT 27.66% | 30/96（31.25%）；GT 15.96% | 完成 |
| chache_val300-36 | YOLOE / yoloe-v8l-seg | [视觉提示 / 10shot / 视觉参考 / 0负](../../runs/chache_val300/YOLOE/yoloe-v8l-seg_10shot/metrics.json) | 10.94 | 4.70 | 1.15 | 5.60 | 17.89 | 57/319（17.87%）；GT 30.32% | 35/124（28.23%）；GT 18.62% | 完成 |

**各方法在本数据集的最高 AP85（仅原总表允许比较的完整实验）**

| 方法 | AP85% | AP90% | AP95% | 四项≤20% AP% | shot / 输入 / 负例 | 实验ID |
|---|---:|---:|---:|---:|---|---|
| GroundingDINO | 7.34 | 6.78 | 1.11 | 9.05 | 0 / text_only / 0 | chache_val300-01 |
| MLLM | 44.61 | 35.27 | 8.75 | — | 10 / O+C / 0 | chache_val300-20 |
| PET-DINO | 13.46 | 12.27 | 3.91 | 13.70 | 1 / 视觉参考 / 0 | chache_val300-27 |
| WeDetect | 25.83 | 20.00 | 3.94 | 28.95 | 0 / text_only / 0 | chache_val300-33 |
| YOLOE | 10.94 | 4.70 | 1.15 | 17.89 | 10 / 视觉参考 / 0 | chache_val300-36 |

**无法计算框误差的实验**

- chache_val300-05 `MLLM提示_v1/qwen3.8-27b_5shot/仅文字提示`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-06 `MLLM提示_v1/qwen3.8-27b_5shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-07 `MLLM提示_v2/qwen3.8-27b_5shot/仅文字提示`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-08 `MLLM提示_v2/qwen3.8-27b_1shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-09 `MLLM提示_v2/qwen3.8-27b_1shot/原图+裁剪图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-10 `MLLM提示_v2/qwen3.8-27b_1shot/原图+裁剪图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-11 `MLLM提示_v2/qwen3.8-27b_1shot/原图+裁剪图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-12 `MLLM提示_v2/qwen3.8-27b_5shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-13 `MLLM提示_v2/qwen3.8-27b_5shot/原图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-14 `MLLM提示_v2/qwen3.8-27b_5shot/原图+裁剪图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-15 `MLLM提示_v2/qwen3.8-27b_5shot/原图+裁剪图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-16 `MLLM提示_v2/qwen3.8-27b_5shot/原图+裁剪图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-17 `MLLM提示_v2/qwen3.8-27b_5shot/原图+裁剪图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-18 `MLLM提示_v2/qwen3.8-27b_5shot/原图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-19 `MLLM提示_v2/qwen3.8-27b_10shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-20 `MLLM提示_v2/qwen3.8-27b_10shot/原图+裁剪图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-21 `MLLM提示_v2/qwen3.8-27b_10shot/原图+裁剪图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-22 `MLLM提示_v2/qwen3.8-27b_10shot/原图+裁剪图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-23 `MLLM提示_v2/qwen3.8-flash_1shot/原图+带框图_无负例`：缺少保存的评估指标/类别范围。
- chache_val300-24 `MLLM提示_v2/qwen3.8-flash_1shot_第2次/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-25 `MLLM提示_v2/qwen3.8-flash_5shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- chache_val300-26 `MLLM提示_v2/qwen3.8-flash_10shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。

## diaowu_99

共 45 组；保存 AP85 的 45 组；完成框误差统计的 0 组。

| # | 方法/模型 | prompt / shot / 输入 / 负例 | AP85% | AP90% | AP95% | AP85:95% | 四项≤20% AP% | score≥0.5 四项≤20% | score≥0.7 四项≤20% | 状态 |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| diaowu_99-01 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 0shot / text_only / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/metrics.json) | 39.16 | 29.09 | 1.98 | 23.41 | — | 26/29（89.66%）；GT 47.27% | 26/29（89.66%）；GT 47.27% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-02 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 0shot / text_only / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/metrics.json) | 38.71 | 30.12 | 2.61 | 23.82 | — | 28/30（93.33%）；GT 50.91% | 28/30（93.33%）；GT 50.91% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-03 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 1shot / O+B / 1负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 38.65 | 28.18 | 2.33 | 23.05 | — | 44/58（75.86%）；GT 80.00% | 44/58（75.86%）；GT 80.00% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-04 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 1shot / O+C / 1负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 30.25 | 22.05 | 3.47 | 18.59 | — | 45/74（60.81%）；GT 81.82% | 45/74（60.81%）；GT 81.82% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-05 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 1shot / O+C+B / 1负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 36.95 | 22.33 | 2.48 | 20.58 | — | 42/55（76.36%）；GT 76.36% | 42/55（76.36%）；GT 76.36% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-06 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 1shot / O+C+B+J / 1负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 33.13 | 22.44 | 0.88 | 18.82 | — | 51/67（76.12%）；GT 92.73% | 51/67（76.12%）；GT 92.73% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-07 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 1shot / O+C+J / 1负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 30.95 | 21.50 | 0.91 | 17.79 | — | 50/81（61.73%）；GT 90.91% | 50/81（61.73%）；GT 90.91% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-08 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 1shot / O+J / 1负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 27.38 | 23.66 | 0.84 | 17.29 | — | 51/86（59.30%）；GT 92.73% | 51/86（59.30%）；GT 92.73% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-09 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+B / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 43.56 | 41.07 | 6.67 | 30.44 | — | 47/58（81.03%）；GT 85.45% | 47/58（81.03%）；GT 85.45% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-10 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+B / 5负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 44.03 | 29.63 | 5.45 | 26.37 | — | 43/50（86.00%）；GT 78.18% | 43/50（86.00%）；GT 78.18% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-11 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+B+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.51 | 18.67 | 4.24 | 20.14 | — | 46/58（79.31%）；GT 83.64% | 46/58（79.31%）；GT 83.64% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-12 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+B+J / 5负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 40.64 | 29.79 | 5.02 | 25.15 | — | 45/59（76.27%）；GT 81.82% | 45/59（76.27%）；GT 81.82% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-13 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 46.20 | 36.62 | 4.67 | 29.16 | — | 46/63（73.02%）；GT 83.64% | 46/63（73.02%）；GT 83.64% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-14 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C / 5负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.51 | 29.24 | 4.32 | 23.69 | — | 42/58（72.41%）；GT 76.36% | 42/58（72.41%）；GT 76.36% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-15 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+B / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 35.95 | 21.90 | 2.61 | 20.15 | — | 44/59（74.58%）；GT 80.00% | 44/59（74.58%）；GT 80.00% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-16 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+B / 5负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 44.83 | 35.50 | 1.87 | 27.40 | — | 44/56（78.57%）；GT 80.00% | 44/56（78.57%）；GT 80.00% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-17 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+B+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 43.97 | 20.11 | 3.76 | 22.61 | — | 44/57（77.19%）；GT 80.00% | 44/57（77.19%）；GT 80.00% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-18 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+B+J / 5负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 41.78 | 26.24 | 2.16 | 23.39 | — | 49/63（77.78%）；GT 89.09% | 49/63（77.78%）；GT 89.09% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-19 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 41.57 | 22.47 | 1.67 | 21.90 | — | 44/60（73.33%）；GT 80.00% | 44/60（73.33%）；GT 80.00% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-20 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+J / 5负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 43.27 | 25.25 | 3.42 | 23.98 | — | 48/61（78.69%）；GT 87.27% | 48/61（78.69%）；GT 87.27% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-21 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 41.35 | 27.34 | 4.98 | 24.56 | — | 51/69（73.91%）；GT 92.73% | 51/69（73.91%）；GT 92.73% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-22 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+J / 5负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 36.21 | 20.91 | 2.48 | 19.87 | — | 41/55（74.55%）；GT 74.55% | 41/55（74.55%）；GT 74.55% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-23 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+B / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 40.23 | 30.29 | 3.48 | 24.67 | — | 41/49（83.67%）；GT 74.55% | 41/49（83.67%）；GT 74.55% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-24 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+B+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 38.47 | 26.24 | 5.33 | 23.34 | — | 45/61（73.77%）；GT 81.82% | 45/61（73.77%）；GT 81.82% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-25 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+C / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 34.62 | 24.75 | 5.83 | 21.73 | — | 41/58（70.69%）；GT 74.55% | 41/58（70.69%）；GT 74.55% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-26 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+C+B / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.39 | 30.44 | 4.60 | 24.14 | — | 45/55（81.82%）；GT 81.82% | 45/55（81.82%）；GT 81.82% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-27 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+C+B+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 43.14 | 25.72 | 3.77 | 24.21 | — | 48/58（82.76%）；GT 87.27% | 48/58（82.76%）；GT 87.27% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-28 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+C+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 38.47 | 22.07 | 3.98 | 21.50 | — | 44/61（72.13%）；GT 80.00% | 44/61（72.13%）；GT 80.00% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-29 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 41.42 | 25.74 | 2.72 | 23.29 | — | 41/55（74.55%）；GT 74.55% | 41/55（74.55%）；GT 74.55% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-30 | MLLM / qwen3.8-flash | [MLLM提示_v1 / 0shot / text_only / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/metrics.json) | 34.05 | 27.80 | 2.72 | 21.53 | — | 24/28（85.71%）；GT 43.64% | 24/28（85.71%）；GT 43.64% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-31 | MLLM / qwen3.8-flash | [MLLM提示_v1 / 5shot / O+B / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 32.01 | 23.73 | 3.78 | 19.84 | — | 41/76（53.95%）；GT 74.55% | 41/76（53.95%）；GT 74.55% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-32 | MLLM / qwen3.8-flash | [MLLM提示_v1 / 5shot / O+B+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 25.18 | 20.57 | 5.12 | 16.96 | — | 46/77（59.74%）；GT 83.64% | 46/77（59.74%）；GT 83.64% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-33 | MLLM / qwen3.8-flash | [MLLM提示_v1 / 5shot / O+C / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 32.62 | 26.29 | 1.37 | 20.09 | — | 42/77（54.55%）；GT 76.36% | 42/77（54.55%）；GT 76.36% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-34 | MLLM / qwen3.8-flash | [MLLM提示_v1 / 5shot / O+C+B / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 34.14 | 28.76 | 4.90 | 22.60 | — | 42/74（56.76%）；GT 76.36% | 42/74（56.76%）；GT 76.36% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-35 | MLLM / qwen3.8-flash | [MLLM提示_v1 / 5shot / O+C+B+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 32.75 | 20.85 | 0.87 | 18.15 | — | 42/78（53.85%）；GT 76.36% | 42/78（53.85%）；GT 76.36% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-36 | MLLM / qwen3.8-flash | [MLLM提示_v1 / 5shot / O+C+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 31.23 | 28.83 | 7.58 | 22.55 | — | 44/80（55.00%）；GT 80.00% | 44/80（55.00%）；GT 80.00% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-37 | MLLM / qwen3.8-flash | [MLLM提示_v1 / 5shot / O+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 25.26 | 22.89 | 1.43 | 16.53 | — | 45/79（56.96%）；GT 81.82% | 45/79（56.96%）；GT 81.82% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-38 | MLLM / qwen3.8-max | [MLLM提示_v1 / 0shot / text_only / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/metrics.json) | 33.47 | 28.81 | 6.08 | 22.79 | — | 21/21（100.00%）；GT 38.18% | 21/21（100.00%）；GT 38.18% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-39 | MLLM / qwen3.8-max | [MLLM提示_v1 / 5shot / O+B / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 31.91 | 20.10 | 0.85 | 17.62 | — | 25/27（92.59%）；GT 45.45% | 25/27（92.59%）；GT 45.45% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-40 | MLLM / qwen3.8-max | [MLLM提示_v1 / 5shot / O+B+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 36.97 | 7.90 | 0.10 | 14.99 | — | 24/26（92.31%）；GT 43.64% | 24/26（92.31%）；GT 43.64% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-41 | MLLM / qwen3.8-max | [MLLM提示_v1 / 5shot / O+C / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 27.60 | 7.89 | 0.00 | 11.83 | — | 30/32（93.75%）；GT 54.55% | 30/32（93.75%）；GT 54.55% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-42 | MLLM / qwen3.8-max | [MLLM提示_v1 / 5shot / O+C+B / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 23.23 | 4.57 | 0.12 | 9.31 | — | 26/27（96.30%）；GT 47.27% | 26/27（96.30%）；GT 47.27% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-43 | MLLM / qwen3.8-max | [MLLM提示_v1 / 5shot / O+C+B+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 25.63 | 7.21 | 0.20 | 11.01 | — | 21/24（87.50%）；GT 38.18% | 21/24（87.50%）；GT 38.18% | 未完成/失败；缺坐标；数量沿用已存统计 |
| diaowu_99-44 | MLLM / qwen3.8-max | [MLLM提示_v1 / 5shot / O+C+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 23.52 | 3.02 | 0.81 | 9.12 | — | 26/31（83.87%）；GT 47.27% | 26/31（83.87%）；GT 47.27% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_99-45 | MLLM / qwen3.8-max | [MLLM提示_v1 / 5shot / O+J / 0负](../../runs/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 30.78 | 9.21 | 0.00 | 13.33 | — | 21/23（91.30%）；GT 38.18% | 21/23（91.30%）；GT 38.18% | 完成（元数据）；缺坐标；数量沿用已存统计 |

**各方法在本数据集的最高 AP85（仅原总表允许比较的完整实验）**

| 方法 | AP85% | AP90% | AP95% | 四项≤20% AP% | shot / 输入 / 负例 | 实验ID |
|---|---:|---:|---:|---:|---|---|
| MLLM | 46.20 | 36.62 | 4.67 | — | 5 / O+C / 0 | diaowu_99-13 |

**无法计算框误差的实验**

- diaowu_99-01 `MLLM提示_v1/qwen3.8-27b_1shot/仅文字提示`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-02 `MLLM提示_v1/qwen3.8-27b_5shot/仅文字提示`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-03 `MLLM提示_v1/qwen3.8-27b_1shot/原图+带框图_含1张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-04 `MLLM提示_v1/qwen3.8-27b_1shot/原图+裁剪图_含1张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-05 `MLLM提示_v1/qwen3.8-27b_1shot/原图+裁剪图+带框图_含1张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-06 `MLLM提示_v1/qwen3.8-27b_1shot/原图+裁剪图+带框图+标注JSON_含1张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-07 `MLLM提示_v1/qwen3.8-27b_1shot/原图+裁剪图+标注JSON_含1张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-08 `MLLM提示_v1/qwen3.8-27b_1shot/原图+标注JSON_含1张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-09 `MLLM提示_v1/qwen3.8-27b_5shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-10 `MLLM提示_v1/qwen3.8-27b_5shot/原图+带框图_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-11 `MLLM提示_v1/qwen3.8-27b_5shot/原图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-12 `MLLM提示_v1/qwen3.8-27b_5shot/原图+带框图+标注JSON_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-13 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-14 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-15 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-16 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+带框图_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-17 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-18 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+带框图+标注JSON_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-19 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-20 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+标注JSON_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-21 `MLLM提示_v1/qwen3.8-27b_5shot/原图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-22 `MLLM提示_v1/qwen3.8-27b_5shot/原图+标注JSON_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-23 `MLLM提示_v1/qwen3.8-27b_10shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-24 `MLLM提示_v1/qwen3.8-27b_10shot/原图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-25 `MLLM提示_v1/qwen3.8-27b_10shot/原图+裁剪图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-26 `MLLM提示_v1/qwen3.8-27b_10shot/原图+裁剪图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-27 `MLLM提示_v1/qwen3.8-27b_10shot/原图+裁剪图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-28 `MLLM提示_v1/qwen3.8-27b_10shot/原图+裁剪图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-29 `MLLM提示_v1/qwen3.8-27b_10shot/原图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-30 `MLLM提示_v1/qwen3.8-flash_5shot/仅文字提示`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-31 `MLLM提示_v1/qwen3.8-flash_5shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-32 `MLLM提示_v1/qwen3.8-flash_5shot/原图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-33 `MLLM提示_v1/qwen3.8-flash_5shot/原图+裁剪图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-34 `MLLM提示_v1/qwen3.8-flash_5shot/原图+裁剪图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-35 `MLLM提示_v1/qwen3.8-flash_5shot/原图+裁剪图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-36 `MLLM提示_v1/qwen3.8-flash_5shot/原图+裁剪图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-37 `MLLM提示_v1/qwen3.8-flash_5shot/原图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-38 `MLLM提示_v1/qwen3.8-max_5shot/仅文字提示`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-39 `MLLM提示_v1/qwen3.8-max_5shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-40 `MLLM提示_v1/qwen3.8-max_5shot/原图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-41 `MLLM提示_v1/qwen3.8-max_5shot/原图+裁剪图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-42 `MLLM提示_v1/qwen3.8-max_5shot/原图+裁剪图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-43 `MLLM提示_v1/qwen3.8-max_5shot/原图+裁剪图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-44 `MLLM提示_v1/qwen3.8-max_5shot/原图+裁剪图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_99-45 `MLLM提示_v1/qwen3.8-max_5shot/原图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。

## diaowu_val300

共 33 组；保存 AP85 的 33 组；完成框误差统计的 10 组。

| # | 方法/模型 | prompt / shot / 输入 / 负例 | AP85% | AP90% | AP95% | AP85:95% | 四项≤20% AP% | score≥0.5 四项≤20% | score≥0.7 四项≤20% | 状态 |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| diaowu_val300-01 | GroundingDINO / grounding-dino-base | [prompt_v1 / 0shot / text_only / 0负](../../runs/diaowu_val300/GroundingDINO/grounding-dino-base_0shot_prompt_v1/metrics.json) | 0.04 | 0.02 | 0.00 | 0.02 | 0.28 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| diaowu_val300-02 | GroundingDINO / grounding-dino-base | [prompt_v2 / 0shot / text_only / 0负](../../runs/diaowu_val300/GroundingDINO/grounding-dino-base_0shot_prompt_v2/metrics.json) | 0.08 | 0.05 | 0.00 | 0.05 | 0.55 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| diaowu_val300-03 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 0shot / text_only / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/metrics.json) | 36.27 | 25.65 | 4.17 | 22.03 | — | 190/230（82.61%）；GT 54.29% | 190/230（82.61%）；GT 54.29% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-04 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 1shot / O+B / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 33.79 | 22.85 | 3.75 | 20.13 | — | 200/245（81.63%）；GT 57.14% | 200/245（81.63%）；GT 57.14% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-05 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 1shot / O+B / 1负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 36.36 | 25.20 | 4.17 | 21.91 | — | 201/238（84.45%）；GT 57.43% | 201/238（84.45%）；GT 57.43% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-06 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 1shot / O+C / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 34.63 | 22.16 | 4.15 | 20.32 | — | 199/243（81.89%）；GT 56.86% | 199/243（81.89%）；GT 56.86% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-07 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 1shot / O+C / 1负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 35.24 | 23.76 | 4.67 | 21.22 | — | 202/244（82.79%）；GT 57.71% | 202/244（82.79%）；GT 57.71% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-08 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+B / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 36.94 | 23.52 | 3.30 | 21.25 | — | 200/229（87.34%）；GT 57.14% | 200/229（87.34%）；GT 57.14% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-09 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+B / 5负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.98 | 26.84 | 2.48 | 22.43 | — | 198/225（88.00%）；GT 56.57% | 198/225（88.00%）；GT 56.57% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-10 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+B+J / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 38.15 | 20.88 | 3.34 | 20.79 | — | 199/234（85.04%）；GT 56.86% | 199/234（85.04%）；GT 56.86% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-11 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+B+J / 5负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 36.02 | 23.95 | 2.83 | 20.93 | — | 201/230（87.39%）；GT 57.43% | 201/230（87.39%）；GT 57.43% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-12 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 33.69 | 21.03 | 3.73 | 19.48 | — | 200/236（84.75%）；GT 57.14% | 200/236（84.75%）；GT 57.14% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-13 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C / 5负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 36.55 | 23.56 | 6.58 | 22.23 | — | 197/221（89.14%）；GT 56.29% | 197/221（89.14%）；GT 56.29% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-14 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+B / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 35.82 | 21.98 | 3.06 | 20.29 | — | 200/230（86.96%）；GT 57.14% | 200/230（86.96%）；GT 57.14% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-15 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+B / 5负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.70 | 22.76 | 3.35 | 21.27 | — | 194/217（89.40%）；GT 55.43% | 194/217（89.40%）；GT 55.43% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-16 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+B+J / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 34.19 | 21.04 | 2.24 | 19.16 | — | 202/239（84.52%）；GT 57.71% | 202/239（84.52%）；GT 57.71% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-17 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+B+J / 5负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 35.98 | 21.86 | 3.97 | 20.60 | — | 198/228（86.84%）；GT 56.57% | 198/228（86.84%）；GT 56.57% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-18 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+J / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 34.41 | 19.92 | 3.14 | 19.16 | — | 202/235（85.96%）；GT 57.71% | 202/235（85.96%）；GT 57.71% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-19 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+C+J / 5负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 35.56 | 19.29 | 2.85 | 19.24 | — | 192/220（87.27%）；GT 54.86% | 192/220（87.27%）；GT 54.86% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-20 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+J / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 31.76 | 22.20 | 2.97 | 18.98 | — | 200/235（85.11%）；GT 57.14% | 200/235（85.11%）；GT 57.14% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-21 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 5shot / O+J / 5负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 36.03 | 22.84 | 3.79 | 20.89 | — | 198/229（86.46%）；GT 56.57% | 198/229（86.46%）；GT 56.57% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-22 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+B / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 35.44 | 22.97 | 3.61 | 20.67 | — | 191/219（87.21%）；GT 54.57% | 191/219（87.21%）；GT 54.57% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-23 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+B / 10负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB10%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 36.47 | 24.97 | 3.41 | 21.62 | — | 186/207（89.86%）；GT 53.14% | 186/207（89.86%）；GT 53.14% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-24 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+C / 0负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.28 | 24.59 | 6.19 | 22.69 | — | 197/222（88.74%）；GT 56.29% | 197/222（88.74%）；GT 56.29% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-25 | MLLM / qwen3.8-27b | [MLLM提示_v1 / 10shot / O+C / 10负](../../runs/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E5%90%AB10%E5%BC%A0%E8%B4%9F%E4%BE%8B/metrics.json) | 37.31 | 25.29 | 3.59 | 22.06 | — | 189/208（90.87%）；GT 54.00% | 189/208（90.87%）；GT 54.00% | 完成（元数据）；缺坐标；数量沿用已存统计 |
| diaowu_val300-26 | PET-DINO / pet_dino_swin_t | [视觉提示 / 1shot / 视觉参考 / 0负](../../runs/diaowu_val300/PET-DINO/pet_dino_swin_t_1shot/metrics.json) | 8.68 | 5.25 | 1.33 | 5.09 | 15.65 | 8/8（100.00%）；GT 2.29% | 0/0（—%）；GT 0.00% | 完成 |
| diaowu_val300-27 | PET-DINO / pet_dino_swin_t | [视觉提示 / 5shot / 视觉参考 / 0负](../../runs/diaowu_val300/PET-DINO/pet_dino_swin_t_5shot/metrics.json) | 2.27 | 1.33 | 0.17 | 1.26 | 5.56 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| diaowu_val300-28 | PET-DINO / pet_dino_swin_t | [视觉提示 / 10shot / 视觉参考 / 0负](../../runs/diaowu_val300/PET-DINO/pet_dino_swin_t_10shot/metrics.json) | 4.40 | 2.93 | 0.43 | 2.59 | 9.42 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| diaowu_val300-29 | WeDetect / wedetect_base | [文字提示_v1 / 0shot / text_only / 0负](../../runs/diaowu_val300/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/wedetect_base_0shot/metrics.json) | 0.99 | 0.01 | 0.00 | 0.33 | 1.00 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| diaowu_val300-30 | WeDetect / wedetect_base | [文字提示_v2 / 0shot / text_only / 0负](../../runs/diaowu_val300/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v2/wedetect_base_0shot/metrics.json) | 0.07 | 0.02 | 0.00 | 0.03 | 0.07 | 0/0（—%）；GT 0.00% | 0/0（—%）；GT 0.00% | 完成 |
| diaowu_val300-31 | YOLOE / yoloe-v8l-seg | [视觉提示 / 1shot / 视觉参考 / 0负](../../runs/diaowu_val300/YOLOE/yoloe-v8l-seg_1shot/metrics.json) | 15.56 | 6.14 | 0.63 | 7.44 | 37.60 | 123/179（68.72%）；GT 35.14% | 103/136（75.74%）；GT 29.43% | 完成 |
| diaowu_val300-32 | YOLOE / yoloe-v8l-seg | [视觉提示 / 5shot / 视觉参考 / 0负](../../runs/diaowu_val300/YOLOE/yoloe-v8l-seg_5shot/metrics.json) | 14.61 | 5.76 | 0.26 | 6.88 | 39.04 | 145/229（63.32%）；GT 41.43% | 112/157（71.34%）；GT 32.00% | 完成 |
| diaowu_val300-33 | YOLOE / yoloe-v8l-seg | [视觉提示 / 10shot / 视觉参考 / 0负](../../runs/diaowu_val300/YOLOE/yoloe-v8l-seg_10shot/metrics.json) | 14.15 | 5.48 | 0.40 | 6.68 | 39.53 | 155/252（61.51%）；GT 44.29% | 132/195（67.69%）；GT 37.71% | 完成 |

**各方法在本数据集的最高 AP85（仅原总表允许比较的完整实验）**

| 方法 | AP85% | AP90% | AP95% | 四项≤20% AP% | shot / 输入 / 负例 | 实验ID |
|---|---:|---:|---:|---:|---|---|
| GroundingDINO | 0.08 | 0.05 | 0.00 | 0.55 | 0 / text_only / 0 | diaowu_val300-02 |
| MLLM | 38.15 | 20.88 | 3.34 | — | 5 / O+B+J / 0 | diaowu_val300-10 |
| PET-DINO | 8.68 | 5.25 | 1.33 | 15.65 | 1 / 视觉参考 / 0 | diaowu_val300-26 |
| WeDetect | 0.99 | 0.01 | 0.00 | 1.00 | 0 / text_only / 0 | diaowu_val300-29 |
| YOLOE | 15.56 | 6.14 | 0.63 | 37.60 | 1 / 视觉参考 / 0 | diaowu_val300-31 |

**无法计算框误差的实验**

- diaowu_val300-03 `MLLM提示_v1/qwen3.8-27b_5shot/仅文字提示`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-04 `MLLM提示_v1/qwen3.8-27b_1shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-05 `MLLM提示_v1/qwen3.8-27b_1shot/原图+带框图_含1张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-06 `MLLM提示_v1/qwen3.8-27b_1shot/原图+裁剪图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-07 `MLLM提示_v1/qwen3.8-27b_1shot/原图+裁剪图_含1张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-08 `MLLM提示_v1/qwen3.8-27b_5shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-09 `MLLM提示_v1/qwen3.8-27b_5shot/原图+带框图_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-10 `MLLM提示_v1/qwen3.8-27b_5shot/原图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-11 `MLLM提示_v1/qwen3.8-27b_5shot/原图+带框图+标注JSON_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-12 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-13 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-14 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-15 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+带框图_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-16 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+带框图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-17 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+带框图+标注JSON_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-18 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-19 `MLLM提示_v1/qwen3.8-27b_5shot/原图+裁剪图+标注JSON_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-20 `MLLM提示_v1/qwen3.8-27b_5shot/原图+标注JSON_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-21 `MLLM提示_v1/qwen3.8-27b_5shot/原图+标注JSON_含5张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-22 `MLLM提示_v1/qwen3.8-27b_10shot/原图+带框图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-23 `MLLM提示_v1/qwen3.8-27b_10shot/原图+带框图_含10张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-24 `MLLM提示_v1/qwen3.8-27b_10shot/原图+裁剪图_无负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。
- diaowu_val300-25 `MLLM提示_v1/qwen3.8-27b_10shot/原图+裁剪图_含10张负例`：缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算。

## 复算与数据来源

`python scripts/analyze_box_precision.py`；只分析一个数据集可使用 `--dataset diaowu_val300`。

原始预测、GT、metrics 和已有 results_summary 均保持原字节。数据来源和 SHA256 记录在 `analysis.json`。框误差复算要求 GT 清单与运行时数据指纹相符，并验证逐图记录完整性及图像哈希；没有复核远端机器上尚未同步的原始预测。
