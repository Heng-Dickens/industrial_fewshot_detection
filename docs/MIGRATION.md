# 原地重组说明

2026-10-26将 `data/processed/`、`data/supports/` 实际合并到 `data/references/`。各场景的参考素材和固定shot清单集中保存，旧地址继续兼容；详见[参考目录说明](guides/REFERENCE_LAYOUT.md)。

工作目录从 `D:\Learn\VLM\annotationtool` 原地重命名为 `D:\Learn\VLM\industrial_fewshot_detection`，未新建副本。重新在编辑器／终端打开新根目录后继续工作。

| 原路径 | 当前实现／数据 |
|---|---|
| `annotate/*.py` | `src/industrial_fewshot/` 的 runner、config、detectors、data、evaluation 等模块 |
| `annotate/tests` | `tests/` |
| `config/*.json` | 共享设置在 `configs/common/`，专用JSON与实验YAML在 `configs/specific/` |
| `prompts` | `src/industrial_fewshot/prompts/templates` |
| `data/DiaoWuexample_prepared` | `data/references/diaowu` |
| `data/DiaoGouexample` | `data/references/diaogou` |
| `data/Chacheexampledata` | `data/references/chache/review` |
| `runs` | `outputs/archive/runs_legacy` |
| `tool` 的完整实现 | `src/industrial_fewshot`，新入口在 `scripts` |
| `annotate` 中的指南 | `docs/guides` |

`annotate` 和 `tool` 保留少量转发入口，兼容以前的导入及命令，不再存放第二份实现。旧 `--config config/diaowu_ablation_config.json` 由位置兼容解析到 `configs/specific`；建议新命令使用 README 中的 YAML。第三方 YOLOE 源码保持原始目录结构，不混入项目业务代码。

配置现只分 `configs/common/` 与 `configs/specific/` 两类；原 models、datasets、experiments、legacy、credentials 目录已合并或分类迁移。MLLM三个实验YAML仍独立设置模型、地址和参数，不继承 `configs/specific/qwen_mllm.yaml`。本地文字检测及JSTY的模型参数已合并到专用入口；YOLOE视觉实验直接继承共享JSON，修改该文件仍影响各shot实验。旧配置路径通过入口映射继续兼容，详见 [配置说明](../configs/README.md)。历史运行记录不回写。

此前尝试的直连默认配置已撤销；当前 MLLM YAML 保持原模型、本地地址及生成参数，但已显式改为 `api_protocol: anthropic_messages`，实际请求 `/v1/messages`，走 CC Switch Claude 分组当前供应商（实测 Aliyun copy），而不是此前返回401的 Codex／千问云 chat/completions 路径。项目使用 `VLLM_API_KEY`／`EMPTY` 规则，不要求填写 `.env`，也未修改 Codex 或 CC Switch 的共享配置。已验证新生成的无敏感几何图片检测、JSON反馈及复核；真实数据集实验尚未完成。协议会写入运行报告及缓存签名，避免跨协议误复用。可选 `env_file` 功能保留但当前实验不启用，`configs/specific` 中的原 JSON 保持原样，旧配置默认协议仍是 openai_chat_completions。

批量标注、复核和格式导出工具也使用同一套位置兼容解析。已有配置仍可用 `config/...` 或旧项目的绝对地址传入；迁移后的配置内容应使用当前路径，配置内相对路径仍以实际配置文件所在目录为基准。

项目目录改名后，虚拟环境中的激活脚本、可编辑安装和命令启动器可能仍绑定旧目录。可离线检查和修复本项目的 Windows YOLOE 环境，不下载模型或改动依赖版本：

```powershell
.\.venv-yoloe\Scripts\python.exe scripts/repair_environment.py --dry-run
.\.venv-yoloe\Scripts\python.exe scripts/repair_environment.py
python scripts/check_links.py
```

修复工具只更新 `.venv-yoloe` 中指向旧 `annotationtool` 根目录的引用，并从已安装包的入口元数据重新生成失效启动器。历史运行、原图和参考资产保持原样。

启动前可在准备使用的终端运行 `python scripts/check_environment.py`，检查实际 Python 路径以及 Pillow、NumPy、SciPy、PyYAML、pycocotools 和 OpenAI 客户端能否真正导入。终端前缀不能替代解释器路径核验。需要补齐依赖时使用该解释器的 `python -m pip install -r requirements.txt`。

旧版本将任何 OpenAI 导入异常都提示为“缺少 openai”，也可能掩盖其依赖的导入问题。现在会区分包缺失与依赖导入失败，并显示实际 Python 路径和原始错误。YOLOE 环境安装脚本也会安装公共实验依赖并执行导入检查。

普通 `python main.py --config ...` 推理如果误用了缺少公共依赖的解释器，入口会检查项目 `.venv-yoloe`，验证完整后在终端明确提示并改用该解释器。`--dry-run`、只评估和帮助入口保持原解释器；设置 `INDUSTRIAL_USE_CURRENT_PYTHON=1` 可关闭自动恢复。

环境恢复入口已改为受管理的子进程，替代原先 Windows 下的 `os.execv`。父进程转发中断并在必要时清理自己创建的子进程；显式使用 `.venv-yoloe\Scripts\python.exe` 可直接执行。CLI 为异步请求安装中断取消处理，取消并收拢并发 worker 后关闭连接；中断返回 130，保留已完成记录和成功缓存。鉴权／权限／模型地址错误（401／403／404）会停止整轮，避免把同一错误逐图重试。

历史运行元数据保留当时的绝对地址，用位置映射解析，不重写历史预测。MLLM 参考资料缓存仍使用迁移前的稳定路径身份，物理移动相同文件不应导致请求指纹改变。数据清单重新生成，当前查询集的内容指纹不变。

机械移动明细在 `.planning/restructure/migration.json`。该报告的旧根路径记录的是迁移来源，不表示仍存在另一份项目。项目重命名记录保存在 `.planning/restructure/rename.json`。
