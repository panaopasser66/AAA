# Task: Stage 2 - Kossel Line Local Random Background Sampling Test

## 0. 当前目标

现在先不管模型训练效果，也不做 defect detection。

本阶段只测试一个核心问题：

```text
已经标记出来的 Kossel line 区域，能不能通过局部随机背景采样，让这些线在视觉上更像背景的一部分。
```

也就是说，本阶段输入是：

```text
data.tif
+ line mask
```

输出是：

```text
delined image / background sampled image
```

我先肉眼看效果。

---

# 1. 当前可用文件

当前文件夹里至少有：

```text
data.tif
line_stage1_dataset/
  pseudo_labels/train_label_selected.png
  responses/response_sato.tif
  responses/response_sato.png
```

如果已经做过模型推理，可能还有：

```text
line_stage1_infer_v1/
  line_prob.tif
  line_mask_thr03.png
  line_mask_thr05.png
  line_mask_thr07.png
```

但本阶段不依赖训练结果。  
如果没有 `line_stage1_infer_v1`，就先用：

```text
line_stage1_dataset/pseudo_labels/train_label_selected.png
```

作为 mask 来源。

---

# 2. 本阶段不要做的事

不要做：

```text
1. 不要训练模型；
2. 不要做 defect detection；
3. 不要做 contrast comparison；
4. 不要输出 defect candidates；
5. 不要修改原始 data.tif；
6. 不要把整张图做全局均值；
7. 不要把 line 区域直接设为 0；
8. 不要直接使用全部 ignore 区域作为 inpaint mask。
```

本阶段只做：

```text
line mask 区域的局部随机背景采样 / patch-like sampling / soft blending。
```

---

# 3. 新建脚本

请新建：

```text
stage2_background_sampling.py
```

运行方式：

```bash
python stage2_background_sampling.py --input data.tif --dataset line_stage1_dataset --out stage2_bg_sampling_test
```

如果有模型推理结果，也支持：

```bash
python stage2_background_sampling.py --input data.tif --dataset line_stage1_dataset --line-prob line_stage1_infer_v1/line_prob.tif --out stage2_bg_sampling_test
```

---

# 4. 输入 mask 的生成逻辑

## 4.1 从 selected label 读取

读取：

```text
line_stage1_dataset/pseudo_labels/train_label_selected.png
```

像素含义：

```text
0   = background
1   = clear line
255 = ignore
```

不要直接把全部 `255 ignore` 都作为 line mask。  
ignore 太宽，会把太多背景也修掉。

请生成几种 mask variant：

## Mask A：clear only

```python
mask_A = label == 1
```

## Mask B：clear + high-confidence ignore

需要读取：

```text
responses/response_sato.tif
```

逻辑：

```python
mask_B = (label == 1) OR ((label == 255) AND (sato_norm > percentile(sato_norm, 90)))
```

## Mask C：clear + medium-confidence ignore

```python
mask_C = (label == 1) OR ((label == 255) AND (sato_norm > percentile(sato_norm, 85)))
```

## Mask D：clear + wider ignore but not full ignore

```python
mask_D = (label == 1) OR ((label == 255) AND (sato_norm > percentile(sato_norm, 80)))
```

如果提供了 `--line-prob`，额外生成：

## Mask E：model probability threshold 0.5

```python
mask_E = line_prob > 0.5
```

## Mask F：model probability threshold 0.3

```python
mask_F = line_prob > 0.3
```

---

# 5. mask 后处理

对每个 mask：

```text
1. 去掉小孤立区域：area < 8 px；
2. 轻微 dilation：disk(1) 或 disk(2)；
3. 不要大幅膨胀；
4. 输出 mask coverage；
5. coverage 超过 20% 时 warning。
```

输出：

```text
stage2_bg_sampling_test/masks/<mask_name>.png
stage2_bg_sampling_test/overlays/<mask_name>_overlay.png
```

---

# 6. protect mask：先保护 compact anomaly

本阶段不是 defect detection，但为了避免把疑似小缺陷直接修掉，需要生成一个简单 protect mask。

目标：

```text
小而紧凑的异常点先不要被背景采样覆盖。
```

建议方法：

```python
local_bg = gaussian_filter(img, sigma=15)
residual = img - local_bg
abs_residual = abs(residual)
candidate = abs_residual > percentile(abs_residual, 99.3)
```

连通域过滤：

```text
area: 4~500 px
eccentricity < 0.9
aspect_ratio < 4
```

生成：

```text
protect_mask.png
protect_overlay.png
```

最终实际修复 mask：

```python
inpaint_mask = line_mask AND NOT protect_mask
```

注意：

```text
protect_mask 不是 defect 检测结果，只是防止第一轮背景采样把 compact blob 抹掉。
```

---

# 7. 核心背景采样方法

请不要再使用简单 local mean hard replacement。  
本阶段要测试 **随机背景采样**。

推荐第一版实现：`random offset background sampling`

## 7.1 基本思想

对每个 inpaint mask 像素 `(y, x)`：

```text
从附近随机选择一个 offset (dy, dx)
取源位置 (y + dy, x + dx)
如果源位置在图像内，并且不是 line / inpaint / protect 区域，
则用该源像素值填充当前 mask 像素。
```

这样做的好处：

```text
1. 采样来自附近区域；
2. 保留背景纹理；
3. 不会像均值一样形成灰色水泥带；
4. 多次采样后可以取 median，降低随机噪声。
```

---

## 7.2 offset 采样要求

参数：

```bash
--num-samples 16
--offset-min 8
--offset-max 80
--source-exclude-dilate 4
--soft-sigma 1.5
```

每次 sampling：

```python
angle = random uniform [0, 2pi]
radius = random uniform [offset_min, offset_max]
dy = round(radius * sin(angle))
dx = round(radius * cos(angle))
```

源像素必须满足：

```python
source_valid = inside_image
               AND NOT source_exclude_mask
```

其中：

```python
source_exclude_mask = binary_dilation(inpaint_mask OR protect_mask, disk(source_exclude_dilate))
```

如果某个像素多次采样都找不到 valid source：

```text
fallback 使用局部 median 或 gaussian background。
```

---

# 8. 多次随机重建

对每个 mask variant，做多次随机重建：

```text
K = num_samples，例如 16
```

得到：

```text
filled_1, filled_2, ..., filled_K
```

最后：

```python
background_fill_median = median(filled_1 ... filled_K)
background_fill_std = std(filled_1 ... filled_K)
```

输出：

```text
background_median.tif
background_uncertainty.png
```

---

# 9. Soft blending

不要硬替换边界。  
使用 soft mask 融合：

```python
soft = gaussian_filter(inpaint_mask.astype(float), sigma=soft_sigma)
soft = clip(soft, 0, 1)

delined = img * (1 - soft) + background_median * soft
```

保存：

```text
delined.tif
delined_preview.png
removed.png
uncertainty.png
```

其中：

```python
removed = img - delined
```

---

# 10. 需要跑多个参数方案

本阶段目标是看哪种采样视觉效果最好，所以请自动跑多个 variant。

至少跑这些组合：

## mask variants

```text
A_clear_only
B_clear_sato90
C_clear_sato85
D_clear_sato80
```

如果有 line_prob：

```text
E_prob05
F_prob03
```

## sampling variants

```text
S1: offset_min=8,  offset_max=40,  num_samples=8,  soft_sigma=1.0
S2: offset_min=10, offset_max=60,  num_samples=16, soft_sigma=1.5
S3: offset_min=20, offset_max=80,  num_samples=16, soft_sigma=2.0
S4: offset_min=30, offset_max=120, num_samples=24, soft_sigma=2.0
```

组合输出，例如：

```text
A_clear_only__S1
A_clear_only__S2
...
D_clear_sato80__S4
```

总数约：

```text
4 masks × 4 sampling = 16 variants
```

如果有 model mask，则更多。

---

# 11. 输出目录结构

输出：

```text
stage2_bg_sampling_test/
  masks/
  overlays/
  variants/
    A_clear_only__S1/
      mask.png
      inpaint_mask.png
      delined.tif
      delined_preview.png
      removed.png
      uncertainty.png
      summary_panel.png
      metrics.json
    A_clear_only__S2/
      ...
  reports/
    stage2_contact_sheet.png
    stage2_summary.csv
```

---

# 12. 每个 variant 的 summary_panel

每个 variant 输出：

```text
summary_panel.png
```

至少包含：

```text
1. input_preview
2. mask_overlay
3. protect_overlay
4. inpaint_mask_overlay
5. delined_preview
6. removed
7. uncertainty
8. zoom crop around strong line/junction if possible
```

---

# 13. 总 contact sheet

生成：

```text
reports/stage2_contact_sheet.png
```

每个 variant 一格，显示：

```text
delined_preview
```

标题包含：

```text
variant name
mask coverage
mean abs removed
uncertainty mean
```

再生成一个更有用的：

```text
reports/stage2_compare_sheet.png
```

每个 variant 显示三列：

```text
mask_overlay | delined_preview | removed
```

这样方便人工快速判断：

```text
1. line 有没有变淡；
2. 有没有水泥灰带；
3. removed 里是不是主要是 line；
4. 交叉点是否糊掉；
5. 背景纹理是否自然。
```

---

# 14. 简单 metrics

每个 variant 输出 `metrics.json`，并汇总到：

```text
reports/stage2_summary.csv
```

字段：

```text
variant_name
mask_name
sampling_name
mask_coverage_percent
inpaint_coverage_percent
mean_abs_removed
std_removed
mean_uncertainty
p95_uncertainty
```

不需要复杂评价，先辅助筛选即可。

---

# 15. 我会怎么人工判断

我会重点看：

```text
stage2_bg_sampling_test/reports/stage2_compare_sheet.png
```

判断标准：

```text
1. Kossel line 是否明显变淡；
2. 线是否变成背景纹理的一部分，而不是灰色水泥带；
3. 强交叉点有没有被糊成一大片；
4. removed 图里主要是不是 line；
5. 背景有没有出现随机盐粒噪声；
6. 小 compact blob 有没有被保护住；
7. uncertainty 高的区域是否集中在难修复线交叉点，而不是整张图都高。
```

---

# 16. 命令

第一轮运行：

```bash
python stage2_background_sampling.py --input data.tif --dataset line_stage1_dataset --out stage2_bg_sampling_test
```

如果已有模型预测：

```bash
python stage2_background_sampling.py --input data.tif --dataset line_stage1_dataset --line-prob line_stage1_infer_v1/line_prob.tif --out stage2_bg_sampling_test
```

---

# 17. 程序结束时打印

```text
Stage 2 background sampling done.
Please check:
  stage2_bg_sampling_test/reports/stage2_contact_sheet.png
  stage2_bg_sampling_test/reports/stage2_compare_sheet.png
  stage2_bg_sampling_test/variants/<best_variant>/summary_panel.png
```
