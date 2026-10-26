# 叉车场景提示词 v1

依据 `data/supports/chache` 中五类、每类10张参考样本编写。类别名沿用当前数据集；这里的中文车型名称用于概括可见外观，不替代原标签。

## 每类中文描述

| 标签 | 简练描述 |
|---|---|
| Chache1 | 普通座驾叉车，驾驶位上方有护顶架，驾驶员坐着，前方有高门架和普通成对细长货叉；车身多为绿色或红色。 |
| Chache2 | 低位搬运车，车体小而低，驾驶员站着，前方伸出贴近地面的长双叉，无高起升门架和护顶架。 |
| Chache3 | 紧凑起重机车，车体较小，驾驶员站着，前方有明显竖直高门架及下部货叉，无驾驶位护顶架。 |
| Chache4 | 带大型专用属具的座驾叉车，驾驶员坐着，带护顶架和粗壮门架，前端装有醒目的黄色宽大搬运属具；车身通常为橙色。 |
| Qianyinche | 单座位牵引车，车体低矮紧凑、外形圆润，带座椅和靠背，驾驶员坐着，无起升门架和成对货叉；车身通常为绿色。 |

主要区分：Chache1看座姿驾驶位、护顶架与普通货叉；Chache2看站姿驾驶位和低位长双叉；Chache3看站姿驾驶位、高门架且无护顶架；Chache4看座姿驾驶位与大型黄色前端属具；Qianyinche看单座驾驶位且无门架、货叉。颜色仅辅助，不能仅因前部不可见就归为牵引车。坐姿、站姿描述对应车型的驾驶布局，无人驾驶也应检测。Chache4的门架不要求为黄色，黄色特征位于前端属具。旧参考配置里的Chache5不作为当前数据集标签。

## 英文短提示

供 GroundingDINO、YOLOE 文本模式等使用。每次取对应类别的一条；现有单类调用不要直接拼接五条。

| 标签 | 英文提示 |
|---|---|
| Chache1 | a sit-down forklift with an overhead guard, a tall front mast and two standard slender forks, usually green or red. |
| Chache2 | a small low-profile stand-on pallet truck with two long low forks, no tall lifting mast and no overhead guard. |
| Chache3 | a compact stand-on lifting truck with a tall vertical mast, lower lifting forks and no overhead guard. |
| Chache4 | a large sit-down forklift with an overhead guard, a sturdy mast and a wide yellow front handling attachment, usually orange. |
| Qianyinche | a compact single-seat tow tractor with a rounded green body, a seat back and no lifting mast or forks. |

## MLLM完整提示词

文件：[chache.txt](../../src/industrial_fewshot/prompts/templates/chache.txt)。包含上述五类对照、部分遮挡处理、整车框选范围及项目所需JSON字段。

模板中的 `{{label_lines}}`、`{{label_name}}` 由项目 `render_prompt` 替换为本次检测标签；MLLM允许在同一次请求中列出五类。五类对照用于判别；只输出配置允许的类别。检测范围为车辆与相连门架、货叉或属具，不为司机、货物或拖车扩大框。

对应的视觉提示位于 data/supports/chache（本地路径 `../../data/supports/chache`），每类 `preview.jpg` 可用于对照检查。

旧版原始模板保存在 [chache_v0_original.txt](../../src/industrial_fewshot/prompts/templates/chache_v0_original.txt)。本次实验的v1短版采用上面的用户修改；v2加长版保存在 [chache_v2_expanded.txt](../../src/industrial_fewshot/prompts/templates/chache_v2_expanded.txt)，补充逐类排除、驾驶姿态、遮挡及框选细则。两版共享类别定义，不依据验证集成绩调整定义。

## 本次对比实验

- v1实验快照：[chache_v1_short.txt](../../src/industrial_fewshot/prompts/templates/chache_v1_short.txt)，与同步后的chache.txt内容一致。
- 短版配置：[chache_qwen_prompt_comparison.yaml](../../configs/specific/chache_qwen_prompt_comparison.yaml)。
- 加长版配置：[chache_qwen_prompt_comparison_expanded.yaml](../../configs/specific/chache_qwen_prompt_comparison_expanded.yaml)。

模型固定为qwen3.8-27b，通过当前CC Switch Claude分组的`http://127.0.0.1:15721/v1/messages`调用。每张图一次请求同时检测五类，每轮仅纯文字、5shot原图＋带框图两组，共300×2×2=1200张图像任务。5shot按每类5个计，共25组参考原图及其带框图；两轮采用相同seed0参考清单，无负例、无裁剪图、无标注JSON输入和额外复核请求。

完整执行或失败后恢复：

```powershell
.\.venv-yoloe\Scripts\python.exe scripts/run_chache_prompt_comparison.py
```

只检查配置和工作量：

```powershell
.\.venv-yoloe\Scripts\python.exe scripts/run_chache_prompt_comparison.py --dry-run
```

进度保存在`outputs/runtime/chache_prompt_comparison/status.json`，逐图完成情况见各结果目录的`records.jsonl`。结果按`outputs/runs/chache_val300/MLLM提示_vN/qwen3.8-27b_5shot/<输入组合>`保存，每组直接包含prompt.txt、metrics.json、evaluation.txt、反馈和预测。跨版本对比写入`outputs/runs/chache_val300/prompt_comparison.md`。分数阈值0.5、0.7的统计同时保留；AP使用完整置信度排序。

## 加长版的模型与shot对比

两个模型均采用加长版v2和每类1、5、10shot原图＋带框图，无负例；每张图一次请求检测五类。共6组、1800张图像任务。已完成的27B5shot先校验完整记录及请求签名，再直接引用原目录，不额外生成一份相同结果。

配置：[Qwen3.8-27B](../../configs/specific/chache_qwen27b_expanded_shots.yaml)、[Qwen3.8-Flash](../../configs/specific/chache_qwenflash_expanded_shots.yaml)。两份配置都显式列出模型名称和shot档位。统一启动或恢复：

```powershell
.\.venv-yoloe\Scripts\python.exe scripts/run_chache_expanded_shots.py
```

加`--dry-run`可检查六组计划。进度和日志位于`outputs/runtime/chache_expanded_shots/status.json`、`launcher.log`；汇总为`outputs/runs/chache_val300/qwen_expanded_shot_comparison.md`。结果继续按`MLLM提示_v2/<模型>_<shot>shot/原图+带框图_无负例`保存，不重跑纯文字组。收到与请求模型名称不符的响应时停止该模型实验，其他可用模型继续完成。

当前CC Switch默认模型为Max，直接请求`qwen3.8-flash`会被映射为Max。Flash配置通过本项目的`proxy_request_model: claude-sonnet-4-6`选中当前供应商的Sonnet→Flash映射；请求路由名和实际要求的模型名分开保存。实际响应仍严格要求为`qwen3.8-flash`，收到Max会拒绝；不修改CC Switch全局配置。27B不使用这个别名。

需要单独运行一个模型时，可使用独立状态目录：

```powershell
.\.venv-yoloe\Scripts\python.exe scripts/run_chache_expanded_shots.py --model qwen3.8-flash --concurrency 1
```

此时进度为`outputs/runtime/chache_expanded_shots/qwen3.8-flash/status.json`，阶段报告为`outputs/runs/chache_val300/qwen3.8-flash_expanded_shot_progress.md`。不同模型可以分别运行；同一个模型不要同时启动两个批次。各进程结束后再运行不带`--model`的命令，会验证并引用已完成的六组结果，统一更新总表。
