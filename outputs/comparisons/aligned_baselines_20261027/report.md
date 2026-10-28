# 对齐条件的算法对比

每组的图片、类别、支持集/提示词和共享评估参数均经过核验。

| 数据集 | 方法 | 条件 | AP50 | 基线AP50 | 差值（百分点） |
|---|---|---|---:|---:|---:|
| diaowu_val300 | YOLOE | 1-shot | 53.035% | 18.996% | +34.039 |
| diaowu_val300 | YOLOE | 5-shot | 54.810% | 10.007% | +44.803 |
| diaowu_val300 | YOLOE | 10-shot | 55.956% | 15.169% | +40.788 |
| diaowu_val300 | GroundingDINO | 吊物 | 0.487% | 1.171% | -0.684 |
| diaowu_val300 | GroundingDINO | 被吊起的圆筒吊物 | 0.997% | 0.071% | +0.926 |
| chache_val300 | YOLOE | 1-shot | 28.141% | 19.490% | +8.651 |
| chache_val300 | YOLOE | 5-shot | 27.262% | 20.326% | +6.937 |
| chache_val300 | YOLOE | 10-shot | 29.004% | 22.537% | +6.467 |
| chache_val300 | GroundingDINO | 简短类别词 | 10.585% | 18.028% | -7.443 |
| chache_val300 | GroundingDINO | 具体类别名 | 10.494% | 19.063% | -8.569 |
| chache_val300 | GroundingDINO | 短描述 | 9.279% | 20.696% | -11.417 |
| chache_val300 | GroundingDINO | 长描述 | 3.789% | 35.101% | -31.312 |
