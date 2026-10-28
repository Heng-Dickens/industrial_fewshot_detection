# 最新吊物300张实验

- [全部运行结果总表与总结](results_summary.md)：23组Qwen3.8-27B实验，纯文字及1/5/10shot的参考输入、负例对照，包含0.5/0.7阈值、原始prompt和详细评估链接。
- [CSV表格](results_summary.csv) / [JSON原始指标](results_summary.json)：与总表逐行对应。
- [5shot全部输入组合](MLLM提示_v1/qwen3.8-27b_5shot/comparison_all_modes.md) / [1/5/10shot对照](qwen27b_visual_shot_comparison.md)。
- [纯文字结果反差核查](text_result_sanity_check.md)。

当前测试集300张，其中210张有目标、90张无目标，共350个真值框。数据指纹为 `470b98531e4a34810e7ee42bf250e39c97480e42427c3d84152e29e4d368f3fc`。旧版145个真值框的300张实验不在当前总表中。

本目录的 `MLLM提示_v1` 是当前允许少量遮挡、逐个目标判断的文字版本；版本号按数据集独立编号。纯文字记录虽位于5shot父目录，实际没有参考图片，总表按0shot列出。23组均有300张有效预测；其中4张第二遍复核失败后采用有效首轮预测，已在总表逐条注明。

更新汇总：`.\.venv-yoloe\Scripts\python.exe scripts/summarize_diaowu_results.py`。此命令只整理现有记录，不调用模型，不修改原始指标或预测。
