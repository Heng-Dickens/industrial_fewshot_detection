# 提示词与运行结果对应

结果现在按测试数据集及真实提示词版本分组，见 [结果目录说明](RESULT_LAYOUT.md)。每个版本的 `prompt_version.json` 保存完整文字、复核模板、指纹及说明；主文字或复核规则改变会自动使用另一版本目录，旧指标仍对应旧提示词。

直接打开 outputs/runs/prompt_results.md（本地路径 `../../outputs/runs/prompt_results.md`），即可按结果目录查看实际文字摘要、AP50、AP50:95、评估阈值、Precision／Recall及运行状态。表内链接指向该次运行自己的 `prompt.txt` 和评估报告；完整结构化索引为 prompt_results.json（本地路径 `../../outputs/runs/prompt_results.json`）。新运行完成、缓存复用导出和单独补算指标后会更新索引。

## 每次运行保存什么

| 文件或位置 | 内容 |
|---|---|
| `prompt.txt` | GroundingDINO／YOLOE文字模式的实际 `text_prompt`，或 MLLM 渲染后的主提示词。YOLOE视觉模式／监督检测明确说明无文字提示 |
| `prompt.json` | 实际文字、SHA256、来源、模型、请求指纹、参考输入模式及数量、参考清单指纹、复核模板；YOLOE另记录可用的视觉参考信息 |
| `run.json`、`summary.json` 的 `targets[].prompt_snapshot` | 新运行的同一份提示词快照；历史恢复另见 `prompt.json` |
| `evaluation.txt` | GroundingDINO／YOLOE文字模式直接显示本次实际文字；其它模型说明提示类型和快照位置 |
| `review_template.txt` | MLLM复核模板；具体首轮和复核输出仍保存在逐图反馈中 |

MLLM 的 `prompt.txt` 保存主文字规则，参考图、参考标注JSON及每张待测图片的消息不展开拼入这个文件。参考材料组合由结果目录名称、`run.json` 的实验信息和 `prompt.json.reference_inputs` 对应。YOLOE视觉模式使用参考原图和框生成视觉原型，文字模式只编码实际描述；JSTY监督模型只输入待测图像。视觉模式和监督模式说明文件里的中文不是实际送入模型的提示词。

## 已有记录恢复

已完成历史运行的提示词恢复。目录整理后，现存记录及新实验以自动更新的索引为准。GroundingDINO历史对比如下：

| 运行 | 实际文字 | AP50 | AP50:95 |
|---|---|---:|---:|
| 第一轮（旧目录未保留，数值见历史对比） | hanging industrial mold. suspended object. | 3.20% | 2.48% |
| 精简描述，当前保留记录（本地路径 `../../outputs/runs/diaowu_99/文字提示_v1/grounding-dino-tiny_0shot/prompt.txt`） | a large hollow cylindrical load suspended from a yellow lifting hook by two taut slings spread to opposite sides, including the hook and slings. | 5.33% | 3.03% |

旧轮数值见 历史结构化对比（本地路径 `../../outputs/reports/groundingdino_prompt_comparison_20261022.json`）。不要仅凭目录后缀推断 prompt；以该目录的 `prompt.json` 为准。

历史恢复只使用该次运行保存的模型参数或原有提示词文件，核验MLLM提示词与原指纹；不会读取当前YAML去覆盖旧轮文本。未保存且无法核验的文字明确标记缺失。恢复过程不修改预测、`metrics.json`、原推理记录、缓存或数值；原评估文本表格保留，只增加提示词说明。

需要再次恢复或更新索引时，在项目根目录执行：

```powershell
# 仅检查，不修改记录
.\.venv-yoloe\Scripts\python.exe scripts/backfill_run_prompts.py --dry-run

# 恢复可核验的提示词记录、更新对照表，不调用模型
.\.venv-yoloe\Scripts\python.exe scripts/backfill_run_prompts.py
```

不要手动将结果目录中的 `prompt.txt` 改成另一个版本；它是该次推理的历史输入证据。要做新实验，请修改模型配置中的实际提示词，再运行对应命令；新结果会保存到另一个运行目录。导出快照和索引不改变推理指纹，不会使已有缓存失效。
