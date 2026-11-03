# 第三步：10shot模型对比补齐

仅比较10shot/类：叉车O+C+B（v2），吊物O+B+J（v1）；27B已有结果复用，补Flash与Max各两个场景。
共4组、1200张次；无负例、seed0，沿用27B相同参考图SHA、提示、生成与评估参数。
等待当前27B吊物1shot和20shot全部完成并汇总后，再做真实模型/图片探测及正式推理。
Flash沿用Sonnet路由，Max沿用原qwen3.8-max请求名；响应必须核验为相应Qwen模型，路由不符即停止，不改代理全局设置。
叉车两模型依次运行，并发2；吊物两模型并行，各并发2。每组完成自动更新子表、CSV和全局汇总。

控制器状态：`running`。

[前置实验进度](../shot_supplement_20261103/README.md)

| 场景 | 模型 | shot/类 | 输入 | 状态 | 成功/总图 | AP50 | AP50:95 | 结果 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| chache | qwen3.8-flash | 10 | O+C+B | completed | 300/300 | 78.24 | 53.53 | [查看](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-flash_10shot_第3次/原图+裁剪图+带框图_无负例/evaluation.txt) |
| chache | qwen3.8-max | 10 | O+C+B | completed | 300/300 | 76.42 | 43.64 | [查看](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-max_10shot/原图+裁剪图+带框图_无负例/evaluation.txt) |
| diaowu | qwen3.8-flash | 10 | O+B+J | completed | 300/300 | 56.78 | 44.17 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-flash_10shot_第3次/原图+带框图+标注JSON_无负例/evaluation.txt) |
| diaowu | qwen3.8-max | 10 | O+B+J | completed | 300/300 | 56.04 | 44.38 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-max_10shot/原图+带框图+标注JSON_无负例/evaluation.txt) |

累计成功：1200/1200张次。

[最终Qwen表：第三步](../../../_shared/comparisons/final_results/01_qwen_inputs.md#step-3)

命令：`python scripts/run_qwen_model10_supplement.py --prepare`（仅离线准备）→ `--background`（等待后自动启动）。
进度：`python scripts/run_qwen_model10_supplement.py --status`。

- 复用chache qwen3.8-27b：[原始评估](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-27b_10shot/原图+裁剪图+带框图_无负例/evaluation.txt)
- 复用diaowu qwen3.8-27b：[原始评估](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-27b_10shot_第4次/原图+带框图+标注JSON_无负例/evaluation.txt)
