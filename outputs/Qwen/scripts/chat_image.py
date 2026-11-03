"""Standalone image-chat demo. Not part of the annotation/test pipeline."""

def main():
    from openai import OpenAI
    import base64
    from pathlib import Path

    # 注意：此处将 client 的 base_url 指向你的 vLLM 服务器地址
    # vLLM 服务通常不校验 API Key，可以随意填写[reference:6][reference:7]
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://172.10.60.169:8000/v1"
    )

    # 1. 纯文本对话
    # chat_completion = client.chat.completions.create(
    #     model="/data/datasets/Qwen/Qwen3-VL-8B-Instruct",
    #     messages=[
    #         {"role": "user", "content": "你好，请介绍一下自己"}
    #     ]
    # )
    # print(chat_completion.choices[0].message.content)


    # 读取本地图片并转换为 Base64（相对本文件定位，不依赖当前工作目录）
    IMAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "0000010.jpg"
    with open(IMAGE_PATH, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 2. 多模态对话 (通过图片 URL)
    chat_completion = client.chat.completions.create(
        model="/data/datasets/Qwen/Qwen3-VL-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "帮我把图片中的绿色牵引车进行标注,把标注完的坐标给我"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
    )
    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main()
