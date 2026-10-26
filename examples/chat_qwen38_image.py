#!/usr/bin/env python3
"""
Qwen3.8-27B-NVFP4 本地 vLLM 传图片调用示例
- 支持本地图片文件（自动转 base64）
- 支持网络图片 URL
- 关 thinking + 限长
"""
import json
import base64
import urllib.request

# ============ 配置 ============
BASE_URL = "https://wytvllm.iepose.cn/v1"      # 本机；内网改 http://192.168.3.43:8000/v1
MODEL = "Qwen3.8-27B-FP8"
MAX_TOKENS = 512
# ==============================

def _image_to_data_url(image_path: str) -> str:
    """本地图片文件 → data URL"""
    import mimetypes
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def chat_with_image(prompt: str, image_path: str = None, image_url: str = None) -> str:
    """
    传图 + 文字
    - 本地图片: image_path="C:/path/图片.png"
    - 网络图片: image_url="https://example.com/图片.jpg"
    - 两者都传 -> 多张图
    """
    content = []
    if image_path:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}})
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append({"type": "text", "text": prompt})

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,               # ② 低温度：减少随机漂移，更锁死确定任务
        "top_p": 0.9,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # 本地图片示例（有测试图就用，没有就报错提示）
    test_img = r"E:\杆塔倒伏和杆塔倾斜\杆塔本体倾斜\with_gtbtqx\images\2023051609585223.jpg"
    print(f"请求: {MODEL} (thinking=OFF, max_tokens={MAX_TOKENS})\n")
    print("【本地图片】")
    prompt = (
        "你是一名输电线路缺陷检测助手。请仔细分析图片，判断杆塔是否存在倾斜。"
        "只输出一个JSON对象，不要输出任何其他文字，格式如下：\n"
        '{"tilted": true/false, "confidence": 0~1, '
        '"bbox": [x1,y1,x2,y2]  # 若有倾斜，杆塔在图片中的归一化边界框0~1}'
    )
    try:
        out = chat_with_image(prompt, image_path=test_img)
        print(out)
    except FileNotFoundError:
        print(f"(测试图 {test_img} 不存在，跳过后再测本地图)")

    print("\n【网络图片示例】")
    print("用法: chat_with_image('描述这张图', image_url='https://...')")
    print("或  : chat_with_image('描述这两张图', image_path='a.png', image_url='https://...')")