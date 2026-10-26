# 实验专用凭据

`configs/common`用于可选的独立凭据方案。当前 MLLM 实验已恢复原 CC Switch 转发及 EMPTY 占位值，不需要此文件。只有自行在配置中明确指定 `api_key_env: QWEN_MLLM_API_KEY` 和 `api_key_file` 时，才使用对应环境变量或`configs/common`下的 `qwen_mllm.local.json`：

```json
{"api_key": "填写你自己的千问云 API Key"}
```

环境变量优先。`configs/common`的 `*.local.json` 已被 Git 忽略；密钥不会写入运行摘要、配置快照或日志。不要把真实密钥填入实验 YAML。

创建这个文件只影响本项目，不需要切换 CC Switch，也不修改 Codex 的配置。缺少密钥时实际推理会明确报错，`--dry-run` 仍可检查数据和实验矩阵。
