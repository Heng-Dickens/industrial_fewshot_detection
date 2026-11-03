# 实验输出目录

结果按方法归档，统一入口为 [outputs/README.md](../../outputs/README.md)。

```text
outputs/
  YOLOE/
    README.md
    experiments/<数据集>/<模型与shot>/
    comparisons/<数据集或专题>/
    finetune/
      README.md
      scripts/{chache,diaowu,common}/
      <数据集>/
        runs/<训练实验>/
        comparisons/<微调对照>/
  PET-DINO/
  WeDetect/
  GroundingDINO/
  Qwen/
    experiments/<数据集>/<提示词版本>/<模型与shot>/<输入组合>/
    comparisons/<数据集>/
    scripts/
  OVD-MLLM/
    experiments/<数据集>/<流程版本>/{source,baseline,filtered,deduplicated,refined}/
    comparisons/<对照实验>/
    scripts/
  JSTY-YOLO/  # 历史吊物监督检测模型
  _shared/
    datasets/<数据集>/      # 数据身份及跨方法总表
    comparisons/           # 需要所有方法共同参与的核查
    runtime/               # 请求缓存、配置、清单副本及日志
    archive/               # 历史布局及总表快照
    environment_setup/
```

各方法的 README 汇总本方法的数据集和比较文档。微调权重、训练日志、测试预测与微调对照均放在该方法的 finetune 中。
原来的 examples 和 tests 已移除；实验脚本与配置随方法保存，公共推理和评估代码仍在 src，命令入口仍在 scripts。

## 单次实验内容

普通推理目录保留 run.json、metrics.json、evaluation.txt、prompt.json，以及本地逐图 records.jsonl、预测和反馈。
视觉提示的 shot 表示提供的参考样本数，不代表更新权重；训练参数、checkpoint 和训练后测试见 finetune。
Qwen 的提示词版本、shot、输入组合分别保留，修改条件后建立新实验目录，不覆盖既有结果。
WeDetect 与 GroundingDINO 的提示词版本仍由 .prompts 注册表分配；同一方法的不同提示词不会散到其他方法目录。

## 汇总与复现

在项目根目录运行：

```bash
python scripts/summarize_results.py
python scripts/check_links.py
python outputs/OVD-MLLM/scripts/summarize_experiment.py --audit
python outputs/YOLOE/finetune/scripts/chache/train.py --method yoloe-vp
```

最后一条仅展示训练计划，正式执行沿用 --execute。汇总读取已保存结果，不调用模型，也不重算或覆盖原始指标。
项目内 output_dir 指向 outputs 时，运行器自动按方法、数据集和实验分组；自定义项目外目录沿用原有显式配置。

## 历史路径

迁移没有改写历史 run.json、原始预测、权重或请求回答。旧绝对路径由
[paths.py](../../src/industrial_fewshot/paths.py) 通过 outputs/_shared/layout_migration.json 解析到新目录；
后续脚本和公共配置使用新目录，不创建旧布局的软链接。
数据集和模型文件仍位于 data 与 weights。
