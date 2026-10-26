# 参考素材统一目录

`data/processed` 和 `data/supports` 已实际合并为 `data/references`：

```text
data/references/
  diaowu/
    positive/、negative/           规范参考素材
    1shot/、5shot/、10shot/、25shot/ seed清单及查看副本
    preparation/                  数据准备报告和COCO
  chache/
    Chache1/、Chache2/、Chache3/、Chache4/、Qianyinche/
      positive/                   各类10个目标及标注/裁剪/带框图
      1shot/、5shot/、10shot/       各类嵌套选样清单
    review/                       标注复核的旧类别示例
    source/                       来源图片和标注
  diaogou/                        吊钩6张参考图及同名标注
```

吊物配置的 `reference_image_dir`、`support_manifest_dir` 和 `preparation.supports_dir` 都指向 `data/references/diaowu`，`preparation.prepared_dir` 指向其 `preparation` 子目录。叉车多类别MLLM两项参考路径都指向 `data/references/chache`，单类别本地检测指向对应类别子目录。

固定选样JSON的内容与哈希保留，历史绝对路径通过兼容解析读取新文件。旧路径不需要软链接或另一份素材副本。MLLM、YOLOE视觉/融合、PET-DINO的请求身份按迁移前已发送材料保持稳定；实际换图、换框、换文字或参数仍会改变指纹。

`data/` 继续被Git忽略。服务器单独准备素材，再配置本地路径并生成适合服务器路径的清单，参见[服务器部署说明](GIT_SERVER_DEPLOYMENT.md)。
