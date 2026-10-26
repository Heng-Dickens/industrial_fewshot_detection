# YOLOE 裁剪视觉特征与文字特征融合

用户自行执行实验，在项目根目录运行：

```powershell
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/yoloe_fusion_5shot.yaml
```

独立配置 `configs/specific/yoloe_fusion_5shot.yaml` 直接列出模型、shot 数、文字、两路权重和推理参数；只继承公共数据集位置。模型为现有 YOLOE-11L，文字编码器为已准备的 MobileCLIP-B(LT)，不需要新增下载或 API 配置。

```yaml
yoloe_fusion:
  positive_count: 5
  selection_seed: 0
  visual_weight: 0.75
  text_weight: 0.25
```

输入使用固定 seed=0 的 5 张正例：`0000090.jpg`、`0000189.jpg`、`0001320.jpg`、`0001352.jpg`、`0003939.jpg`。从 prepared 参考数据的 manifest 读取对应目标裁剪图，每张裁剪的全图区域作为视觉提示，不把原图背景或负样本送入编码器。标注来源于独立支持集，不使用待测图片的真值作为提示。原图及标注路径保留用于来源核验和测试集排重。

计算顺序：每张裁剪的视觉特征 L2 归一化；同一原图有多张目标裁剪时先在原图内平均并归一化，再对 5 张原图平均并归一化得到 V。文字经 MobileCLIP 和 YOLOE RepRTA 对齐后归一化得到 T。最后使用 `P = normalize(0.75 * V + 0.25 * T)`，设置为一个 `hanging object` 类别的提示原型，再检测待测图像。该加权融合是本项目的实验策略，不将两路提示当成两个类别。

实际文字与最新 GroundingDINO／YOLOE 纯文字实验相同：

```text
a large hollow cylindrical load suspended from a yellow lifting hook by two taut slings spread to opposite sides, including the hook and slings.
```

结果直接保存到 `outputs/runs/diaowu_99/文字提示_v1/yoloe-11l-seg_5shot_裁剪图0.75+文字0.25/`，重跑使用同级 `_第2次` 等后缀。保存原有 COCO/VOC 评估、预测与反馈，`prompt.txt` 保留实际英文；`prompt.json` 记录文字、两路权重、参考原图身份、各裁剪路径及内容指纹，并更新根目录 `prompt_results.md`。

修改文字、裁剪内容、两路权重或模型权重会使旧推理缓存失效；只修改评估阈值可以复用框。沿用现有固定 99 张测试图，模型候选阈值 0.001、NMS IoU 0.7、输入 640，评估 IoU 0.5、工作点分数阈值 0.5。不要用测试集结果调权重后再将其当作独立评估。

只检查配置和输入、暂不推理：

```powershell
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/yoloe_fusion_5shot.yaml --dry-run --report-format text
```

配置准备期间仅做了 dry-run 和模拟模型验证，没有代替用户运行该融合实验。实际效果以用户执行后生成的报告为准。
