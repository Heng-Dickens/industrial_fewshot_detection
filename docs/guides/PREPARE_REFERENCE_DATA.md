# 整理标注参考数据

`prepare_reference_data.py` 读取图片目录及图片旁的同名 LabelMe JSON，将正负样本和不同参考输入材料分开保存。全部复制到新输出目录，源图片、JSON 和已有裁剪图保持不变。不调用模型。

在项目根目录执行：

```powershell
python scripts/prepare_references.py --input-dir "D:\dataset\references" --dry-run
python scripts/prepare_references.py --input-dir "D:\dataset\references"
```

默认输出在原目录的同级 `<原目录名>_prepared`，例如 `data/DiaoWuexample_prepared`。也可以指定位置、包含子目录：

```powershell
python scripts/prepare_references.py --input-dir "D:\dataset\references" --output-dir "D:\dataset\prepared" --recursive
```

## 正负样本规则

- 图片有同名有效 LabelMe JSON，且 shapes 中有目标：正样本。
- 没有同名 JSON，或 JSON 的 shapes 是空数组：负样本，沿用当前数据集约定。
- 已存在但损坏的 JSON、缺少 shapes、无效类别/坐标、越界或零面积框、尺寸不符：校验时报错，不把它们当作负样本。
- 支持 rectangle 和 polygon。polygon 的局部裁剪和可见标注框使用其外接矩形，原 JSON 中的多边形点不变。其它 shape 类型报错。
- 默认排除文件名含 `_crop_数字_类别` 的裁剪图和以 `_boxed` 结尾的带框图，避免把这些派生正样本错分成负样本；`--include-derived` 可显式纳入。

脚本先校验所有输入，再开始导出。输出不能位于输入目录内部，也不会覆盖已有非空输出目录；再次导出时请指定一个新目录。

## 输出结构

```text
DiaoWuexample_prepared/
  positive/
    images/                     # 正样本原图，逐字节复制
    annotations/                # 对应 JSON
    crops/<类别目录>/           # 每个目标一张局部 PNG
    boxed_images/               # 原分辨率的彩色框全图 PNG，附类别名称
  negative/
    images/                     # 无标注或空标注的原图
    annotations/                # 仅保留输入中原本存在的空标注 JSON
  classes.json                  # 类别、颜色、裁剪目录、图片/目标数量
  manifest.json                 # 总数、正负划分、来源哈希、框坐标和导出路径
```

开启 recursive 后，会递归读取来源目录，但输出不保留来源子文件夹：images、annotations、boxed_images 下面直接是文件，crops 仅按类别分目录。通常保留原文件名；不同来源存在同名图片时，自动加入稳定的短哈希后缀，并同步 JSON、裁剪图及带框图名称。来源路径只记录在 manifest.json。同一来源目录内不同扩展名却共用同一 JSON 文件名的图片仍会报错。

导出的 JSON 保留 shapes、类别、flags 等标注内容，将 imagePath 改为相对于导出 JSON 的对应 images 路径，清空重复的 imageData；源 JSON 不变。无 JSON 的负样本不伪造原始标注。

每个目标独立裁剪，按类别放入 crops 子目录，文件名如 `0000090_crop_001_hanging_object.png`。浮点框的左上角向下取整、右下角向上取整，覆盖完整标注范围。裁剪和全图预览不缩放；PNG 有透明通道时保留透明背景。

同一类别始终使用同一种颜色，同批数据不同类别分配不同颜色。类别按名称排序，首类为红色；颜色表和目录名写入 classes.json，类别名称包含不适用于 Windows 路径的字符时自动转换并处理重名。

可用 `--line-width 5` 调整框线宽度，`--font-size 20` 调整类别字大小，`--font "C:\Windows\Fonts\msyh.ttc"` 指定中文字体。

## 用作标注模型的参考数据

现有参考读取模块能识别包含该 manifest 的整理目录。将场景配置 reference_image_dir 指向 `<原目录名>_prepared` 后，JSON 示例只读 positive/annotations，裁剪示例只读 positive/crops，负例只读 negative/images；不会将分开存放的正样本原图或 boxed_images 误当成负例。

本脚本只准备输入素材，不改变 main 的参考输入组合或启动推理。`positive/boxed_images` 是供后续带框全图参考输入使用的素材目录。
