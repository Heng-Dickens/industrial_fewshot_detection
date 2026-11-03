# 原 99 张吊物测试集

实际数据和真值指纹见 [dataset.json](dataset.json)。[prompt_results.md](prompt_results.md) 对照各模型的提示词与指标。

- `MLLM提示_v1`：原严格可见规则，保留 Qwen 不同 shot 和参考材料组合。
- `文字提示_v1`：GroundingDINO、YOLOE文字与融合模式的实际英文描述。
- `视觉提示`：YOLOE 的图像参考实验。
- `监督检测`：JSTY本地监督模型。

模型目录内直接保存结果，Qwen对比实验再按参考组合分组。新的300张验证集独立放在 [diaowu_val300](../diaowu_val300/README.md)。
