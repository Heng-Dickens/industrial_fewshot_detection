# Chache特种车辆实验结果

**统一入口：[所有运行结果总表与总结](results_summary.md)**。每个实际运行占一行，包括成功、未完成及失败记录；可点击查看该次prompt和详细评估。

- [CSV表格](results_summary.csv)：UTF-8 BOM，适合Excel筛选；`_pct`列为百分数。
- [JSON明细](results_summary.json)：原始0～1指标、类别AP50、数据指纹和结果来源。
- [扩展版5shot全部组合及前三名shot对比](qwen27b_expanded_full5shot_top3.md)。
- [短版与扩展版prompt对比](prompt_comparison.md)。
- [27B与Flash的原图＋带框图shot对比](qwen_expanded_shot_comparison.md)。

数据为最终版300张、188个目标框。每类shot独立计数，所有参考均为正例。纯文字实际是0shot；历史目录中的5shot名称不代表纯文字输入了参考图。

各模型目录中的原`comparison.txt`只覆盖该次启动配置的组合，并非整个场景的总表。原始结果不移动、不删除；统一表从单组记录读取，避免重复计入汇总文件。

更新总表（不调用模型）：

```powershell
.\.venv-yoloe\Scripts\python.exe scripts/summarize_chache_results.py
```
