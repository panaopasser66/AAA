# Project2 Task: Line-band Contrast Suppression v1

## 0. 现在重新定义目标

当前目标不是 defect detection，也不是复杂 inpainting。

现在只做一件事：

```text
把肉眼能看到的黑色 Kossel line 视觉上藏进背景里。
```

换句话说：

```text
对于已经 label / mask 出来的线区域，
不要再当成“缺洞”让模型猜整块背景，
而是把这些线像 PS 修图一样重新赋值：
让线区域的灰度分布接近它两侧的背景，
从而降低 line contrast。
```

核心目标：

```text
黑线不再显眼；
线位置变成局部背景的一部分；
不要出现灰色水泥带；
不要把整张背景抹平；
不要做 defect detection。
```

---

# 1. 思路改变

之前尝试过：

```text
1. pixel-wise random sampling
2. median sampling
3. self-supervised inpainting
```

这些方法的问题是：

```text
1. pixel-wise sampling 不保持线两侧背景连续；
2. median 会把纹理抹平；
3. inpainting 模型容易保守，明显主线仍然残留；
4. 目标其实不是生成全新图像，而是降低已知 line band 的 contrast。
```

现在改成：

```text
line-band contrast suppression
```

也就是：

```text
对每条 Kossel line 的整条带状区域：
1. 估计这条线两侧的局部背景；
2. 计算 line 像素相对于背景的 residual；
3. 把 residual 缩小；
4. 给 line 区域重新赋值为“背景 + 少量纹理残差”；
5. 让黑线视觉上消失或明显变淡。
```

核心公式：

```python
bg = estimated_local_background
residual = img - bg
out[line_core] = bg[line_core] + keep_ratio * residual[line_core]
```

其中：

```text
keep_ratio = 0.0 ~ 0.3
```

解释：

```text
keep_ratio = 0.0：线区域完全贴近背景
keep_ratio = 0.1：只保留 10% 原始 line contrast
keep_ratio = 0.3：保留 30% line contrast
```

---

# 2. 当前文件

当前 project2 中至少有：

```text
data.tif
```

如果已经跑过 prepare，则可能还有：

```text
outputs/prepare/
  responses/response_sato.tif
  masks/mask_A_clear_sato98.png
  masks/mask_B_sato95.png
  masks/mask_C_sato90.png
  masks/mask_D_sato85.png
  masks/protect_mask.png
```

如果没有 `outputs/prepare`，请先运行或自动调用：

```bash
python scripts/prepare_project2_data.py --input data.tif --out outputs/prepare
```

---

# 3. 新建脚本

请新建：

```text
scripts/line_band_contrast_suppress.py
```

运行方式：

```bash
python scripts/line_band_contrast_suppress.py --input data.tif --prepared outputs/prepare --out outputs/line_band_suppress_v1
```

本阶段不要训练模型。

---

# 4. 输入 mask

从下面这些 mask 读取：

```text
outputs/prepare/masks/mask_A_clear_sato98.png
outputs/prepare/masks/mask_B_sato95.png
outputs/prepare/masks/mask_C_sato90.png
outputs/prepare/masks/mask_D_sato85.png
```

如果文件不存在，请从 `response_sato.tif` 重新生成：

```python
A = sato_norm > p98
B = sato_norm > p95
C = sato_norm > p90
D = sato_norm > p85
```

并保存。

---

# 5. line band mask 后处理

对于每个 mask，生成不同宽度的 line band。

请自动跑这些 mask variant：

```text
B_sato95
C_sato90
D_sato85
```

每个 mask 再跑这些宽度：

```text
width2
width4
width6
width8
```

处理方式：

```python
line_core = mask
line_band = binary_dilation(line_core, disk(width))
line_band = binary_closing(line_band, disk(1 or 2))
```

注意：

```text
line_band 是要被重新赋值的区域。
width 太小会留下黑边；
width 太大会误伤背景。
所以必须输出多版本让我肉眼选。
```

输出：

```text
outputs/line_band_suppress_v1/masks/<variant>_line_band.png
outputs/line_band_suppress_v1/overlays/<variant>_overlay.png
```

---

# 6. protect mask

读取或生成：

```text
outputs/prepare/masks/protect_mask.png
```

如果不存在，生成：

```python
local_bg = gaussian_filter(img, sigma=15)
residual = img - local_bg
candidate = abs(residual) > percentile(abs(residual), 99.3)
```

连通域过滤：

```text
area: 4~500 px
eccentricity < 0.9
aspect_ratio < 4
```

实际处理 mask：

```python
edit_mask = line_band AND NOT protect_mask
```

说明：

```text
protect_mask 不是 defect 检测；
只是暂时不让算法把小而紧凑的异常点直接抹掉。
```

---

# 7. 背景估计方法

请不要用全局均值。  
请不要把 line 区域设成 0。  
请不要直接对整张图高斯模糊后替换。

需要实现 3 种背景估计方法，对比效果。

---

## Method 1: masked Gaussian background

目标：

```text
用 line 外的像素估计低频背景。
```

实现：

```python
valid = ~edit_mask
weight = valid.astype(float)

num = gaussian_filter(img * weight, sigma=bg_sigma)
den = gaussian_filter(weight, sigma=bg_sigma)
bg = num / max(den, eps)
```

测试：

```text
bg_sigma = 8, 15, 25
```

这个方法快，但可能偏平滑。

---

## Method 2: side-band local background

目标：

```text
用 line 两侧附近的背景估计 line 区域应该有的灰度。
```

实现：

```python
inner = binary_dilation(edit_mask, disk(width + 2))
outer = binary_dilation(edit_mask, disk(width + side_radius))
side_band = outer AND NOT inner
side_band = side_band AND NOT edit_mask AND NOT protect_mask
```

然后用 side_band 估计背景：

```python
side_weight = side_band.astype(float)
num = gaussian_filter(img * side_weight, sigma=side_sigma)
den = gaussian_filter(side_weight, sigma=side_sigma)
bg_side = num / max(den, eps)
```

测试：

```text
side_radius = 20, 40, 80
side_sigma = 8, 15
```

这个方法更符合“线两边背景赋值”的直觉。

---

## Method 3: local distribution matching / residual attenuation

目标：

```text
不要生成全新背景，只把 line contrast 压低到局部背景附近。
```

实现：

```python
bg = masked_gaussian_background or side_band_background
residual = img - bg
out_core = bg + keep_ratio * residual
```

测试：

```text
keep_ratio = 0.0, 0.1, 0.2, 0.3
```

这个就是最核心的“重新赋值降低 contrast”。

---

# 8. 可选纹理保留

为了避免 line 区域变成完全平滑灰带，可以加入少量背景纹理。

方法：

```python
texture = img - bg
side_texture_values = texture[side_band]
random_texture = random sample from side_texture_values
out_core = bg + keep_ratio * residual + texture_scale * random_texture
```

测试：

```text
texture_scale = 0.0, 0.25, 0.5
```

注意：

```text
第一版 texture_scale 可以默认 0 或 0.25。
如果随机纹理产生盐粒噪声，就设为 0。
```

---

# 9. 融合方式：core hard + boundary feather

不要再让整个 mask 都 soft blend，容易保留原始黑线。  
请改成：

```text
core 区域：直接用 out_core 替换
边界区域：做 feather blending
非 mask：保留 raw
```

定义：

```python
core = edit_mask
boundary = binary_dilation(edit_mask, disk(feather_radius)) AND NOT core
```

融合：

```python
out = img.copy()
out[core] = out_core[core]

boundary_alpha = distance based alpha, 从 1 平滑降到 0
out[boundary] = alpha * out_core[boundary] + (1 - alpha) * img[boundary]
```

测试：

```text
feather_radius = 1, 2, 4
```

关键要求：

```text
line core 不要保留原始 raw。
否则黑线还会残留。
```

---

# 10. 自动跑多方案

不要只输出一张。  
请自动跑一批方案，但不要太爆炸。

建议第一轮跑：

```text
mask: B_sato95, C_sato90, D_sato85
width: 2, 4, 6
background_method: masked_gaussian, side_band
bg_sigma: 15, 25
keep_ratio: 0.0, 0.1, 0.2
feather_radius: 2
texture_scale: 0.0
```

大约：

```text
3 masks × 3 widths × 2 methods × 2 sigmas × 3 keep_ratios = 108 variants
```

如果太多，可以先跑前 36 个，但请优先包含：

```text
C_sato90_width4_side_sigma15_keep0.1
C_sato90_width6_side_sigma15_keep0.1
D_sato85_width4_side_sigma15_keep0.1
D_sato85_width6_side_sigma15_keep0.1
B_sato95_width4_side_sigma15_keep0.1
```

---

# 11. 每个 variant 输出

输出目录：

```text
outputs/line_band_suppress_v1/variants/<variant_name>/
```

每个 variant 保存：

```text
line_band.png
edit_mask.png
overlay.png
background_estimate.tif
suppressed.tif
suppressed_preview.png
removed.png
summary_panel.png
metrics.json
```

其中：

```python
removed = img - suppressed
```

---

# 12. 总览图

生成：

```text
outputs/line_band_suppress_v1/reports/contact_sheet.png
outputs/line_band_suppress_v1/reports/compare_sheet.png
```

`contact_sheet.png`：

```text
每个 variant 一格，只显示 suppressed_preview
```

`compare_sheet.png`：

```text
每个 variant 一行：
overlay | suppressed_preview | removed
```

这是最重要的，我要用它肉眼挑结果。

---

# 13. metrics

保存：

```text
outputs/line_band_suppress_v1/reports/summary.csv
```

字段：

```text
variant_name
mask_name
width
method
sigma
keep_ratio
feather_radius
edit_coverage_percent
mean_abs_removed_inside
mean_abs_removed_outside
removed_inside_outside_ratio
```

指标只辅助判断，不作为唯一标准。

---

# 14. 人工判断标准

我会重点看：

```text
outputs/line_band_suppress_v1/reports/compare_sheet.png
```

判断：

```text
1. 黑色主线是否明显消失或变淡；
2. line 位置是否接近周围背景；
3. 是否没有灰色水泥带；
4. removed 里主要是不是 line；
5. 交叉点有没有糊成一大片；
6. 小 compact blob 是否没有被全部抹掉；
7. 背景大尺度亮暗是否保留。
```

---

# 15. 运行命令

先准备数据：

```bash
python scripts/prepare_project2_data.py --input data.tif --out outputs/prepare
```

然后运行本脚本：

```bash
python scripts/line_band_contrast_suppress.py --input data.tif --prepared outputs/prepare --out outputs/line_band_suppress_v1
```

---

# 16. Codex 执行提示

你可以对 Codex 说：

```text
请阅读 project2_task_line_band_contrast_suppress_v1.md。

现在我的目标不是 inpainting 生成整块背景，也不是 defect detection。
我要的是：对于已经 label 出来的贯穿 Kossel 黑线区域，重新赋值降低它们相对于周围背景的 contrast，让这些黑线视觉上隐藏进背景。

请新建 scripts/line_band_contrast_suppress.py。
核心公式是：
  bg = local background estimate
  residual = raw - bg
  suppressed[line_core] = bg[line_core] + keep_ratio * residual[line_core]

请自动跑 B/C/D mask、多种宽度、多种 keep_ratio，输出 compare_sheet 让我肉眼选。
```
