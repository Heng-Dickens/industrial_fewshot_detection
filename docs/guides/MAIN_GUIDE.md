# 统一标注入口

`main.py` 是通用入口，通过必填的 `--config` 读取**当前场景的独立配置**。每个场景自行定义模型、目标类别、标注子程序、数据目录、参考图和提示词；入口没有固定场景或目标总表。

```powershell
python -m pip install -r requirements.txt
python main.py --config configs/specific/diaowu_annotation_config.json --dry-run
python main.py --config configs/specific/diaowu_annotation_config.json
python main.py --config configs/specific/diaogou_annotation_config.json
python main.py --config configs/specific/diaowu_annotation_config_2.json --mode train
```

上面三个已有场景配置均可直接使用。先将对应文件的 `image_dir` 改为当前存在的图片目录。以后新增场景时，复制 `configs/common/scene_template.json` 为任意文件，例如 `configs/specific/workshop_b_forklift.json`，修改该文件即可：

```powershell
python main.py --config configs/specific/workshop_b_forklift.json
```

模式也可完全写在场景文件的 `mode` 中：设为 `train` 就标注并评估，设为 `test` 就只标注，无需修改 Python 代码。场景文件可放在任意目录，不需要在入口注册文件名或目标名。

`train` 按本项目约定是“推理并对照原有标注进行评估”，不更新模型权重。`test` 只推理并导出标注，不读取已有 JSON 作为跳过条件，也不计算指标。

`--mode`、`--image-dir`、`--output-dir` 可临时覆盖场景配置；不传则使用配置值。`--limit 5` 仅用于小批调试，正常运行不传，处理目录内所有图片。`--dry-run` 检查配置、真值和参考图，显示实际输出根目录，不调用 API、不创建结果目录。脚本退出码：0 成功，1 配置/数据/运行错误，2 有推理失败。

## 配置

参考材料和张数对比由 `experiment_suite` 配置，按当前启用的材料与正负例开关运行；七种材料、两种负例设置及文字基线全部开启时为15组，当前用户配置只启用5-shot原图＋JSON、无负例及文字基线，共2组。1/10/25张可自行启用。结果按模型与shot数命名，实验组下一层直接保存结果。组合、参数和输出说明见 [REFERENCE_SUITE_GUIDE.md](REFERENCE_SUITE_GUIDE.md)。

所有 JSON 配置中的相对路径均相对于**所选场景配置所在目录**解析；命令行图片和输出路径相对于当前工作目录。模板中的图片、参考图和模型名称是占位值，使用前请替换。

| 字段 | 含义 |
| --- | --- |
| `scene_name` | 场景名称，记录到运行摘要中；省略时使用配置文件名 |
| `mode` | `train` 或 `test` |
| `backend` | VLM 场景策略：`diaowu`、`diaogou`、`generic`；本地检测策略：`yoloe`，使用 yoloe 配置及参考原图/框，详见 [YOLOE_GUIDE.md](../YOLOE_GUIDE.md) |
| `model` / `base_url` | 模型名称和兼容 OpenAI 的服务地址，与原场景配置格式一致；也接受 model 对象中的 name/base_url |
| `api_key_env` / `api_key` | 优先读取环境变量，否则使用配置值。运行快照不保存凭据 |
| `labels` | 当前场景要标注的一个目标，格式 `[{"name":"类别名","description":"目标定义"}]`；类别名须与评估真值一致 |
| `image_dir` | 待标注图片根目录 |
| `ground_truth_dir` | 真值根目录；null 表示图片旁的同名 LabelMe JSON。独立目录按图片相对路径匹配 |
| `recursive` | 是否包含子文件夹，默认 true |
| `missing_ground_truth` | `negative`：按数据集约定，将缺少 JSON 的图片视为无目标负样本，预测框计入误报；`skip`（默认）：仍标注所有图片，缺少真值的图片不计指标；`error`：遇到缺失真值立即停止。空 shapes 的有效 JSON 始终是目标类负样本 |
| `output_dir` | 可选输出根目录；省略、null 或空字符串时使用输入目录的同级目录，名称为输入目录名 + output_suffix；不得放在输入、真值或参考目录内部 |
| `output_suffix` | 默认 `_annotations`；例如输入 `00000` 时输出到同级 `00000_annotations`。可改为 `_predictions` 等文件夹名称后缀 |
| `reuse_predictions` | 默认 true，普通命令自动复用兼容的成功推理结果，只请求新增、变化或失败图片 |
| `cache_version` | 默认 v1；手动修改可让旧推理缓存失效，例如代理上游更换但模型名/地址未变 |
| `reference_image_dir` | 当前场景目标的参考图目录，只读；扫描待标注目录时排除参考目录 |
| `reference_mode` | 默认 configured 保留原流程；text_only、text_json、text_crops、text_json_crops 分别选择四种输入组合，JSON 仅为参考标注文本，不附参考全图 |
| `ablation` | enabled=true 时按 modes 列表运行参考输入对比实验；需 train，输出比较 CSV/JSON |
| `prompt_file` | 匹配当前目标的完整检测提示词文件，修改后直接生效 |
| `review_prompt_file` | 第二遍复核指令；复核同时保留完整检测提示词；省略时使用内置通用复核文件 |
| `max_reference_examples` / `max_reference_crops` / `max_reference_negatives` | 带框正例、正例裁剪图、负例数量上限；0 禁用对应输入 |
| `concurrency`、`max_tokens` 等 | 并发、重试、超时、生成参数、是否复核、输出格式；沿用原有场景配置字段 |
| `evaluation` | Precision/Recall 的 IoU、置信度阈值、每图片每类别最大检测数、代理损失权重 |

三个场景策略现在共用参考图准备和模型请求，均可配置带框正例、裁剪正例和负例。吊物场景通过 `require_suspension_evidence` 保留悬吊确认过滤，并使用现有六张正例小图；目标判定规则由对应 `prompt_file` 决定。新目标使用 `generic` 并填写自己的定义和提示词，不需要新增目标 Python 脚本。新场景应提供该目标的真实参考样例；暂时没有样例时，把三个参考数量上限设为 0，并保留一个存在的空参考目录。

也接受将数据字段放在 `dataset` 中、生成参数放在 `inference` 中的分组写法。分组中的生成参数优先于同名顶层字段。现有扁平场景配置和模板均无需改成这种写法。

提示词文件支持 `{{label_name}}`、`{{label_description}}`、`{{label_lines}}`，复核文件还支持 `{{initial_result}}`。JSON 示例的普通花括号无需转义。吊物、吊钩规则分别在 `src/industrial_fewshot/prompts/templates/diaowu.txt` 和 `src/industrial_fewshot/prompts/templates/diaogou.txt`，代码中不再保留相同规则的副本。

四组输入组合、吊物场景对比配置、运行命令及指标差值说明见 [ABLATION_GUIDE.md](ABLATION_GUIDE.md)。待测图片真值只用于本地评估，模型输入的标注 JSON 来自独立参考样例。

原来的两份 `*_annotate.py` 和两份 `*_batch_folders.py` 已合并到公共模块，不再作为入口。需要处理多个子文件夹时，设置 `image_dir` 为父目录并开启 `recursive: true`。模块与测试职责见 [README.md](../../README.md)。旧配置中只用于旧入口的检查点、跳过原有标注、原地删除旧标注字段已经清理。

## 输出与真值保护

每次运行生成独立目录，不修改或删除原图和原有 JSON，也不会因为图片旁已有 JSON 而跳过推理。

输出根目录的优先级是：命令行 `--output-dir` → 配置中的非空 `output_dir` → 输入目录名加后缀的同级目录。默认路径基于本次实际输入目录，因此 `--image-dir` 覆盖输入后，默认输出位置也随之变化。现有场景文件中显式指定的 `output_dir` 继续生效；要使用默认规则，将该字段删除或设为 null，例如：

```json
{
  "output_dir": null,
  "output_suffix": "_annotations"
}
```

例如输入 `D:\Data\江苏通用\训练数据\模具吊物喷油\模具车间1\00000`，未指定输出时根目录为同级的 `00000_annotations`，其中按模型与 shot 保存完整结果。若输出到项目 `outputs/runs`，会再按数据集与提示词版本分组，见 [结果目录说明](RESULT_LAYOUT.md)：

```text
<实际输出根目录>/<模型与shot>/
  run.json                   # 配置摘要、提示词和参考输入指纹，不含 API key
  records.jsonl              # 完成一张立即记录；含最终框、置信度、回复和失败信息
  summary.json               # 全部请求汇总
  metrics.json               # 仅 train
  prompt.txt                 # 实际主文字提示；视觉/监督模型明确说明无文字提示
  prompt.json                # 提示词快照、来源、文字指纹和参考输入信息
  review_template.txt
  predictions/<原相对目录>/<图片名>.json
```

预测 JSON 为 LabelMe rectangle，`imagePath` 指向原图，shape 附带 `confidence`。无目标或无法判断的成功请求也导出空 `shapes`，失败请求不伪造空标注。运行摘要保留 `scene_name` 和所选配置的绝对路径，方便追溯不同场景。

### 默认自动复用成功推理

普通标注命令默认启用复用，不需要额外参数。再次运行原命令时，先匹配成功记录；一致的图片直接读取已有预测，只请求新增、内容变化、失败或尚未完成的图片。评估阶段失败后，安装好依赖再运行原来的 train 命令，会自动复用推理并重新计算指标。

复用依据是**实际推理输入**：模型名和地址、目标与场景策略、渲染后的提示词和复核提示词、实际选中的参考图/标注内容、生成参数，以及每张图片的 SHA256。改变模型、提示词、参考图、温度或复核设置会触发重新推理；只改并发、重试次数、超时、train/test 模式、评估阈值或真值 JSON，不会重新请求模型。

每次仍创建独立运行目录，保留本次完整预测和评估。跨运行缓存放在 `output_dir/.inference_cache/`，完成一张立即保存，进程中断也可在下次自动复用。已有旧运行会按保存的请求指纹、像素坐标和图片修改时间检查兼容性：旧记录没有图片内容哈希时，只接受未在旧运行后修改的图片；接入新缓存后使用内容哈希。历史成功结果所在的 output_dir 需保留。

切换到默认同级目录时，还会只读查找项目 `runs/<scene_name>` 和旧默认 `../runs` 下的兼容结果；使用 `--output-dir` 覆盖配置时，也会查找原配置中的输出目录和默认同级目录。复用成功后将完整结果及缓存保存到本次选定的目录，旧结果保留。自定义输出目录改到其它位置后，如需复用原结果，请保留其配置路径并通过 `--output-dir` 指定新位置。

日志会先显示“复用成功结果 X 张，待请求模型 Y 张”，`--dry-run` 可以只读查看这两个数字，不创建结果或调用模型。所有结果都命中时不创建模型客户端。同配置并发运行会通过逐图片锁等待并复用另一进程的结果。

运行摘要中的 `reused_prediction_count` 和 `executed_inference_count` 区分复用与本次实际推理；记录的 `cache_hit=true`、`reused_from` 标明来源，缓存记录中的 usage 保留的是历史调用信息。

明确需要重新请求模型时，加 `--force`，或在场景配置中设置 `reuse_predictions: false`。如果 cc-switch 后台切换了供应商，但程序中的模型名和地址都没变，需要修改 `cache_version` 或使用 `--force` 主动刷新。

```powershell
python main.py --config configs/specific/diaowu_annotation_config_2.json --mode train --image-dir "D:\dataset\images"
python main.py --config configs/specific/diaowu_annotation_config_2.json --mode train --image-dir "D:\dataset\images" --dry-run
python main.py --config configs/specific/diaowu_annotation_config_2.json --mode train --image-dir "D:\dataset\images" --force
```

### 推理完成后评估失败：直接补算

`records.jsonl` 和预测 JSON 在每张图片完成后立即保存，评估缺少 NumPy/SciPy 不会丢失它们。`summary.json` 现在也会在评估之前写出，评估失败时记录 `evaluation_status=failed` 和错误原因；新运行会在第一次模型调用前检查评估依赖。

在运行 main.py 的同一个 Python/Conda 环境安装依赖，然后指定原来完整的运行目录：

```powershell
python -m pip install numpy scipy
python main.py --evaluate-only "outputs/archive/runs_legacy/某个场景/原来的运行目录"
```

该命令只读取 `run.json`、`records.jsonl`、原始图片尺寸与真值 JSON，写出同一运行目录里的 `metrics.json` 和 `summary.json`。它不创建模型客户端、不读取参考图、不发起推理，也不改写 `records.jsonl` 或预测 JSON。当前配置里的模型、图片目录或提示词即使改了，也不会触发重新推理：恢复使用原运行记录的图片目录与类别。

新运行会保存评估参数和真值根目录。对于早期只有 `run.json`/`records.jsonl` 的运行，建议显式提供原场景配置，以读取其评估参数和可选的独立真值目录：

```powershell
python main.py --config configs/specific/diaowu_annotation_config_2.json --evaluate-only "outputs/archive/runs_legacy/某个场景/原来的运行目录"
```

补算时，配置中显式设置的 `missing_ground_truth` 会覆盖原运行保存的策略；例如改为 negative 后，可直接把已有完整预测中的无 JSON 图片纳入负样本评估，不需要重新推理。

可加 `--dry-run` 只读检查记录数量、图片和真值，不写出指标。记录不完整（例如计划 100 张、只完成 4 张）会报错，防止仅对部分图片计算出有误导性的指标。`--evaluate-only` 不能与 `--mode`、`--image-dir`、`--output-dir`、`--limit` 或 `--force` 同用。中途推理中断时，直接重新运行普通命令，由自动缓存复用已完成的图片；这里只评估入口用于指定某次完整运行。

## 指标和 loss 的定义

支持 LabelMe 矩形真值，保留原始浮点像素坐标，校验图像尺寸和框范围；其它类别的框不参与本次评估。目标类的 polygon、ignore/difficult/crowd 标记会报错，避免悄悄换算出另一种指标。

- `mAP50`、`mAP75`、`mAP50_95`：按类别、按模型返回的 confidence 降序做一对一 IoU 匹配，101 个 Recall 点插值；IoU 从 0.50 到 0.95、步长 0.05。每图片每类别默认保留最高分 100 个框。无真值的类别 AP 为 null，不参与 mAP 平均。没有任何目标真值时 mAP 为 null，误检仍计入 Precision。

负样本图片（空 shapes 或在 negative 策略下缺少 JSON）也参与 Precision/AP 与代理 loss，预测出的目标框计为 FP。额外输出 `negative_false_positive_image_count`（误报图片数）、`negative_false_positive_box_count`（误报框数）、`negative_true_negative_image_count`（正确拒识数）、`negative_false_positive_rate`（误报图片数 / 成功评估的负样本图片数），以及 `negative_false_positive_images`（误报图片相对路径和框数）。这些图片级指标使用 score_threshold 和每类别检测数量上限。

负样本请求失败单列为 `negative_failed_image_count`，不会冒充正确拒识；没有成功评估的负样本时，负样本误报率为 null。`ground_truth_json_image_count` 统计实际 JSON 数量，`assumed_negative_image_count` 统计按约定无 JSON 的负样本，`ground_truth_image_count` 统计可评估图片总数（包含两者）。

`image_accuracy` 是按全部可评估图片计算的整体正确率：`correct_image_count / evaluated_image_count`。图片必须推理成功，且在配置 IoU/置信度阈值下所有真值均按类别一对一匹配、没有多框或漏框，才计为正确；负样本必须成功返回空预测。任何推理失败都计入 `incorrect_image_count`，保留在分母中，即使它本来没有目标。`failed_image_count` 统计失败图片数，`incorrect_images` 列出错误或失败图片。

`negative_error_rate` 是负样本整体错误率：`(误报图片数 + 推理失败图片数) / 全部负样本图片数`，失败不会被排除。它与仅衡量已完成检测误报的 negative_false_positive_rate 分列。标准检测 Precision 继续按 TP/(TP+FP) 计算，不把未产生预测框的请求失败伪造为误报框；判断实际流程可靠性时同时看 image_accuracy、negative_error_rate 和失败数。
- `precision`、`recall`、`f1`：在配置的 `iou_threshold` 和 `score_threshold` 下计算微平均；同时输出各类别指标和 TP/FP/FN。AP 扫描全部分数，不使用这个固定 score_threshold 截断曲线。
- `mean_matched_iou`：上述固定工作点正确匹配框的平均 IoU。
- `loss.type=detection_proxy_loss`：API 不提供真实训练 loss，因此仅报告可解释的评估代理损失。按图片和类别用 Hungarian 分配匹配框，定位项为归一化 xyxy 的平均 L1，以及平均 `1-IoU`；另加固定工作点的 `FP/(TP+FP)` 和 `FN/(TP+FN)`，四项权重由 `loss_weights` 配置。空分母记 0。`loss.total` 是四项加权和，`is_model_training_loss=false`，不能用于反向传播。

AP 的插值和阈值参考 [COCO 官方评估实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)。这里计算普通矩形、全部面积的 COCO 风格 AP，不包含官方完整评估中的 crowd/ignore/面积分组。VLM 的 confidence 是模型自报分数，未经检测器校准；模型遗漏 confidence 时本次请求报错并按配置重试，不默认塞入 1.0。

推理失败的请求在评估中贡献空预测，真值仍保留，并将 `evaluation_status` 标为 `incomplete`、退出码设为 2。不会只留下成功样本让指标虚高。复核失败保留首轮有效预测，并在记录中保留 `review_error`。
