# Task: Stage 1A - Kossel Line Pseudo-label Variant Sweep

## 0. 当前情况

当前文件夹中已有：

```text
data.tif
prepare_line_dataset.py
line_stage1_dataset/
```

已经运行过第一版：

```bash
python prepare_line_dataset.py --input data.tif --out line_stage1_dataset
```

目前输出中：

```text
line_stage1_dataset/responses/response_sato.png
line_stage1_dataset/overlays/overlay_train_label.png
line_stage1_dataset/reports/patch_contact_sheet.png
```

第一版结果说明：

1. `response_sato.png` 对 Kossel line 响应很好；
2. `overlay_train_label.png` 的 clear line 基本能覆盖强 Kossel line；
3. 但是很多弱线没有进入 ignore，容易被错误当作 background；
4. 当前版本可以作为 v0，但不建议直接训练。

本次任务目标：

```text
不要训练模型。
不要做背景修复。
不要做 defect detection。

只做 pseudo-label 参数 sweep。
输出多套 train_label overlay，让我人工选择哪一套最适合训练 U-Net。
```

---

# 1. 核心原则

pseudo-label 应该分成三类：

```text
0   = background
1   = clear Kossel line
255 = ignore / uncertain
```

训练时：

```text
clear line 用作正样本；
background 用作负样本；
ignore 不参与 loss。
```

本阶段最重要的原则是：

```text
green clear line 宁可少一点，但必须准；
yellow ignore 可以多一点，要覆盖弱线、交叉点、可疑小斑点；
background 必须尽量干净，不要把明显弱 Kossel line 当成 background。
```

---

# 2. 要修改什么

请修改或扩展 `prepare_line_dataset.py`。

新增一个模式：

```bash
python prepare_line_dataset.py --input data.tif --out line_stage1_dataset --label_sweep
```

或者新建独立脚本也可以：

```bash
python sweep_pseudo_labels.py --dataset line_stage1_dataset
```

两种方式都可以，但推荐不要破坏原有功能。

---

# 3. 输入

直接读取已有 response 文件，优先使用：

```text
line_stage1_dataset/responses/response_sato.tif
line_stage1_dataset/responses/response_frangi.tif
line_stage1_dataset/responses/response_coherence.tif
line_stage1_dataset/previews/input_preview.png
```

如果 `.tif` response 不存在，则从 `data.tif` 重新计算 response。

---

# 4. 输出目录

新增目录：

```text
line_stage1_dataset/label_sweep/
  labels/
  overlays/
  masks/
  reports/
```

输出所有候选 label 和 overlay。

---

# 5. 需要生成的 label variants

请至少生成以下 8 个版本。

## Variant A：保守 clear，较保守 ignore

```text
clear = sato > p97.5
loose = sato > p90
ignore = loose - clear
```

输出名：

```text
A_clear97p5_loose90
```

## Variant B：保守 clear，更宽 ignore

```text
clear = sato > p97.5
loose = sato > p85
ignore = loose - clear
```

输出名：

```text
B_clear97p5_loose85
```

## Variant C：更保守 clear，更宽 ignore

```text
clear = sato > p98
loose = sato > p85
ignore = loose - clear
```

输出名：

```text
C_clear98_loose85
```

## Variant D：更保守 clear，很宽 ignore

```text
clear = sato > p98
loose = sato > p80
ignore = loose - clear
```

输出名：

```text
D_clear98_loose80
```

## Variant E：Sato + Frangi + Coherence 组合 ignore

```text
clear = sato > p97.5

loose = (
    sato > p85
    OR frangi > p90
    OR coherence > p90
)

ignore = loose - clear
```

输出名：

```text
E_sato85_frangi90_coh90
```

## Variant F：强线 clear，弱线全部 ignore

```text
clear = sato > p98

loose = (
    sato > p82
    OR frangi > p88
    OR coherence > p88
)

ignore = loose - clear
```

输出名：

```text
F_strong_clear_wide_ignore
```

## Variant G：保护小团状结构，不把 compact blob 当 clear line

```text
clear_raw = sato > p97.5
loose = sato > p85

compact_candidate:
  area 4~500 px
  eccentricity < 0.85
  aspect_ratio < 4

clear = clear_raw AND NOT compact_candidate
ignore = (loose OR compact_candidate) AND NOT clear
```

输出名：

```text
G_compact_protect
```

## Variant H：交叉点和星芒区域进入 ignore

```text
clear = sato > p97.5
loose = sato > p85

junction_candidate:
  high response local maxima
  or connected component area large and multi-orientation
  or line density high after dilation

ignore = (loose OR junction_candidate) AND NOT clear
```

输出名：

```text
H_junction_ignore
```

如果 H 实现复杂，可以先用简化版本：

```text
junction_candidate = binary_dilation(clear, disk(5)) AND sato > p90
ignore = (loose OR junction_candidate) AND NOT clear
```

---

# 6. 形态学后处理要求

对每个 variant：

## 6.1 clear line

```text
1. 去掉太小的孤立区域，例如 area < 8 px；
2. 对 clear line 做轻微 dilation：disk(1)；
3. 不要过度膨胀，避免 green 变成粗带。
```

## 6.2 ignore

```text
1. ignore 可以适当 dilation：disk(1) 或 disk(2)；
2. ignore 需要覆盖 clear line 周围的不确定边界；
3. ignore 也要覆盖明显弱线；
4. ignore 不能覆盖整张图太多，超过 30% 要 warning。
```

## 6.3 label 合成

```python
label = zeros_like(img, dtype=uint8)
label[ignore] = 255
label[clear] = 1
```

注意：

```text
clear 优先级高于 ignore。
如果一个像素同时属于 clear 和 ignore，最终应为 clear=1。
```

---

# 7. 每个 variant 需要输出什么

对每个 variant 输出：

```text
label_sweep/labels/<variant_name>_train_label.png
label_sweep/masks/<variant_name>_clear.png
label_sweep/masks/<variant_name>_ignore.png
label_sweep/overlays/<variant_name>_overlay.png
```

overlay 颜色：

```text
green  = clear line
yellow = ignore
background = 原图灰度
```

overlay 文件上方或文件名中需要标注：

```text
clear ratio
ignore ratio
background ratio
```

例如：

```text
B_clear97p5_loose85_overlay_L3.2_I14.5.png
```

其中：

```text
L = clear line percentage
I = ignore percentage
```

---

# 8. contact sheet

生成总览图：

```text
label_sweep/reports/label_sweep_contact_sheet.png
```

内容：

```text
每个 variant 一格，显示 overlay。
标题写：
variant name
clear ratio
ignore ratio
```

还要生成一个 patch-level contact sheet：

```text
label_sweep/reports/patch_compare_contact_sheet.png
```

要求：

```text
选 16 个代表性 patch：
1. 强线区域
2. 弱线区域
3. 交叉点区域
4. 背景区域
5. 疑似小 compact blob 区域

每个 patch 按列展示 A-H 的 label overlay。
```

如果实现 patch-level 对比比较复杂，至少先输出全图 `label_sweep_contact_sheet.png`。

---

# 9. summary CSV

输出：

```text
label_sweep/reports/label_sweep_summary.csv
```

字段：

```text
variant_name,clear_pixels,clear_percent,ignore_pixels,ignore_percent,background_pixels,background_percent,warning
```

warning 示例：

```text
ignore > 30%, too much ignored area
clear > 12%, clear line may be too broad
clear < 1%, clear line may be too conservative
```

---

# 10. 验收标准

运行：

```bash
python prepare_line_dataset.py --input data.tif --out line_stage1_dataset --label_sweep
```

或者：

```bash
python sweep_pseudo_labels.py --dataset line_stage1_dataset
```

必须输出：

```text
line_stage1_dataset/label_sweep/
  labels/*.png
  masks/*.png
  overlays/*.png
  reports/label_sweep_contact_sheet.png
  reports/label_sweep_summary.csv
```

我会重点看：

```text
line_stage1_dataset/label_sweep/reports/label_sweep_contact_sheet.png
```

选择标准：

```text
1. green clear line 是否主要覆盖强 Kossel line；
2. green 不要覆盖大量 compact 小斑点；
3. yellow ignore 是否覆盖弱 Kossel line；
4. background 是否干净，不要含大量明显弱线；
5. ignore 不要大到整张图都不能训练。
```

---

# 11. 这一步不要做

本阶段不要：

```text
1. 不要训练 U-Net；
2. 不要做 line removal；
3. 不要做 background inpainting；
4. 不要做 defect detection；
5. 不要修改原始 data.tif；
6. 不要只输出一个 label；
7. 不要把 pseudo-label 当作最终 ground truth。
```

---

# 12. 最后打印

程序结束时请打印：

```text
Done label sweep.
Please check:
  line_stage1_dataset/label_sweep/reports/label_sweep_contact_sheet.png
  line_stage1_dataset/label_sweep/reports/label_sweep_summary.csv
```
