# 实验结果

- [全部提示词与结果对照](prompt_results.md)：按数据集、提示词、模型和参考输入组合查看指标及实际文字。
- [Chache特种车辆：全部结果总表与总结](chache_val300/results_summary.md)：集中对照27B、Flash、不同prompt、shot与参考输入，包含完成和失败状态。
- [diaowu_99](diaowu_99/README.md)：原 99 张吊物测试集。
- [最新吊物：全部结果总表与总结](diaowu_val300/results_summary.md)：当前300张、350个真值框，汇总纯文字及1/5/10shot、负例和参考输入组合；[目录说明](diaowu_val300/README.md)。

结果统一放在 `<数据集>/<提示词版本或输入方式>/<模型与shot>/`。普通模型直接记录结果，Qwen对比实验再按参考组合分组。详细规则见 [结果目录说明](../../docs/guides/RESULT_LAYOUT.md)。

提示词版本号在各数据集目录下独立编号，不能跨数据集将同一个v1/v2理解为同一份文字。各版本的 `prompt_version.json` 说明实际文字；各组的 `prompt.txt` / `prompt.json` 与该次指标一一对应。最新吊物300张目录中的 `MLLM提示_v1` 已采用允许少量遮挡、逐个目标判断的规则。

`.inference_cache` 为自动复用的内部缓存。遗留数据在 `outputs/archive/runs_legacy`，中断分组证据在 `outputs/archive/interrupted_runs`，运行设置在 `outputs/runtime`。历史提示词、预测内容和指标保留。
