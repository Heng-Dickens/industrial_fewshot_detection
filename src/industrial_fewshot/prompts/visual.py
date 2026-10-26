"""Paired visual-reference message formatting."""

def build_suite_messages(config: dict, prompt: str, current_url: str) -> list[dict]:
    content = []
    def image(text, asset):
        content.extend([{"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": asset["rendered_data_url"]}}])
    for index, sample in enumerate(config["reference_samples"], 1):
        category = f"，类别 {sample['label']}" if sample.get('label') else ''
        image(f"正样本 {index}{category} 原图：符合目标标注条件。以下同编号的材料均来自此图。", sample["original"])
        for number, crop in enumerate(sample.get("crops", []), 1):
            image(f"正样本 {index} 的目标裁剪图 {number}：学习目标外观与完整组件范围。", crop)
        if "boxed" in sample:
            image(f"正样本 {index} 的可见标注框图：颜色框表示对应类别的人工标注范围。", sample["boxed"])
        if "json" in sample:
            content.append({"type": "text", "text": f"正样本 {index} 的标准 LabelMe JSON（坐标属于该参考原图，"
                            "图像可能经等比例缩小；不能复制为待测图坐标）：\n" + sample["json"]["compact_json"]})
    for index, negative in enumerate(config["reference_negatives"], 1):
        image(f"负样本 {index} 原图：不符合目标标注条件，正确结果为不标注任何框。", negative)
    content.extend([{"type": "text", "text": "只使用本次实际提供的参考信息；提示词提及但未提供的参考类型直接略过。\n"
                    + prompt + "\n以上材料只用于学习目标特征与标注条件。最后一张是唯一待识别全图，"
                    "请搜索全部区域，只返回该全图的目标与坐标；无目标时返回空结果。"},
                    {"type": "image_url", "image_url": {"url": current_url}}])
    return content

