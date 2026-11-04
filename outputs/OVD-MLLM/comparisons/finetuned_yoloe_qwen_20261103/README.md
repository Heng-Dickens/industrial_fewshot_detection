# 三种YOLOE微调→Qwen27B（1shot）

两个场景各选训练10shot、视觉提示10shot（固定头不需要提示）的AP50最佳实际配置，共6组，测试各300张。
Qwen每类仅1张原图+人工带框参考，无负例，seed0。直接复用原YOLOE预测，不重新训练或运行GPU推理。
S0原始预测→B0候选score≥0.01/每类最多100框→B1筛选→B1D同类同实例去重→B2坐标修正。
每请求最多20个候选；无候选不调用Qwen，uncertain保留。筛选禁止补框/改类/改分数；框修正保留候选集合、类别和分数。
输入含原图、编号整图与1.5倍上下文裁图；整图最长边1280、裁图640、JPEG90。旧未微调级联保留作为历史对照。
每场景并发2个图像任务，共最多4；逐请求记录usage及重试，验证成功的请求按完整输入哈希缓存。
格式异常最多尝试3次；重试仅追加格式纠错要求，保持图片、审核标准与候选不变。异常响应单独留档，不作为成功预测。
AP50最佳来自现有测试结果的探索性选择，输入尺寸/seed/训练范围随选中配置保留，不当作等预算独立验证。

控制器状态：`completed`。

| 场景 | 微调方式 | 原始AP50 | 候选框 | 筛选批数 | 状态 | 完成/300 | B1 AP50 | B2 AP50 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| chache | 固定头／移除SAVPE | 82.22 | 445 | 234 | completed | 300 | 63.98 | 62.06 |
| chache | 视觉提示／更新SAVPE | 64.13 | 3250 | 336 | completed | 300 | 43.45 | 50.72 |
| chache | 视觉提示／冻结SAVPE | 59.02 | 3651 | 353 | completed | 300 | 42.27 | 49.97 |
| diaowu | 固定头／移除SAVPE | 92.85 | 505 | 263 | completed | 300 | 57.95 | 56.72 |
| diaowu | 视觉提示／更新SAVPE | 84.98 | 405 | 258 | completed | 300 | 55.09 | 54.85 |
| diaowu | 视觉提示／冻结SAVPE | 83.33 | 487 | 275 | completed | 300 | 55.62 | 55.51 |

已落盘请求尝试2799次（失败19次）；非缓存输入token 13,725,580，缓存输入token 10,015,360，输出token 585,722。仅统计收到usage的请求，不将缓存读入与非缓存输入按同价合计；实际费用以平台账单为准。

[第四部分最终比较表](../../../_shared/comparisons/final_results/04_yoloe_qwen.md)
[前置10shot模型对比](../../../Qwen/comparisons/model10_supplement_20261103/README.md)

命令：`python scripts/run_finetuned_yoloe_qwen.py --prepare`（离线）→ `--background`；`--status`查看。

- chache_head：[选中原始预测](../../../YOLOE/finetune/chache_val300/runs/yoloe_search_20261029_r2/head_full_aug/test_val300/fixed_head/evaluation.txt) · [本组结果](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/experiment_plan.json)
- chache_savpe_train：[选中原始预测](../../../YOLOE/finetune/chache_val300/runs/yoloe_search_20261029_r2/vp_cross_neck/test_val300/visual_10shot/evaluation.txt) · [本组结果](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/experiment_plan.json)
- chache_savpe_frozen：[选中原始预测](../../../YOLOE/finetune/chache_val300/runs/yoloe_chache_frozen_savpe_20261030/vp_cross_neck_frozen_savpe_seed1/test_val300/visual_10shot/evaluation.txt) · [本组结果](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/experiment_plan.json)
- diaowu_head：[选中原始预测](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/head_full_960_seed2/test_val300/fixed_head/evaluation.txt) · [本组结果](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/experiment_plan.json)
- diaowu_savpe_train：[选中原始预测](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/vp_full_lr0.0001/test_val300/visual_10shot/evaluation.txt) · [本组结果](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/experiment_plan.json)
- diaowu_savpe_frozen：[选中原始预测](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_frozen_savpe_20261030/vp_full_lr0.0001_frozen_savpe/test_val300/visual_10shot/evaluation.txt) · [本组结果](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/experiment_plan.json)
