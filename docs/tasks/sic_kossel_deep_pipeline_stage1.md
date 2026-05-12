# SiC X-ray Kossel Line 深度学习处理路线与第一步任务书

## 0. 当前目标

你现在手里只有：

```text
data.tif
```

后续可能会有很多张 `data.tif` 或多张 `.tif` 图像。

当前总思路不再是继续调传统参数，也不再是用硬 mask + 局部均值把线抹掉，而是改成：

```text
1. 深度学习先学会什么是 Kossel line
2. 得到 line probability map / line mask
3. 对 line 区域做局部随机背景采样或背景生成
4. 让线视觉上变成自然背景的一部分
5. 对原图、去线图、预测背景做 contrast comparison
6. 标出 compact defect candidate
```

重要原则：

```text
不要一开始就训练 defect。
先把 Kossel line 学好、遮罩做好、背景重建做好。
defect detection 放到最后做。
```

---

# 1. 总体 Pipeline

## Stage 1：LineMaskNet —— 学习 Kossel line mask

目标：

```text
输入 X-ray tif 图像
输出 Kossel line probability map
```

推荐输入通道：

```text
channel 1: 原始 signed image
channel 2: abs(image)
channel 3: Sato response
channel 4: Frangi / Meijering response
channel 5: structure tensor coherence 或 gradient magnitude
```

推荐模型：

```text
U-Net
ResUNet
DeepLabV3+
```

第一版建议先用轻量 U-Net，不要一开始上很复杂的模型。

输出：

```text
line_prob.png
line_mask_hard.png
line_overlay.png
```

---

## Stage 2：Background Synthesizer —— 把 line 变成自然背景

目标：

```text
line mask 区域不再用均值硬填，而是用附近背景随机采样 / patch-based sampling / inpainting 方式重建。
```

核心思想：

```text
line mask M
→ 周围扩张得到 local background band
→ 排除其它 line 和 compact anomaly
→ 从附近背景中随机采样
→ 多次重建
→ 得到 background_median 和 uncertainty_std
```

输出：

```text
background_pred.tif
delined_image.tif
background_uncertainty.png
delined_preview.png
```

---

## Stage 3：Contrast Comparison —— 找 compact defect candidate

目标：

```text
用原图和预测背景的差异找异常点。
```

推荐计算：

```text
residual = raw - background_pred
z_score = residual / local_noise
```

候选筛选条件：

```text
1. abs(z_score) 足够高
2. 面积在合理范围，例如 4~500 px
3. 形状紧凑
4. eccentricity 不太高
5. aspect ratio 不像长线
6. 多次随机背景重建中稳定存在
7. 不只是 line 交叉点或 inpainting 边界伪影
```

输出：

```text
residual_zscore.png
defect_candidates_overlay.png
defect_candidates.csv
```

---

# 2. 为什么先做 Stage 1

现在只有 `data.tif`，所以第一步不要直接做完整 deep learning pipeline。

第一步应该完成：

```text
从 data.tif 生成 line pseudo-label 和训练 patch，
为后续训练 LineMaskNet 做准备。
```

也就是说，第一步不是直接训练大模型，而是先建立数据结构：

```text
data.tif
→ 预处理
→ 计算 Sato / Frangi / Meijering / coherence
→ 生成 pseudo line label
→ 生成 ignore 区域
→ 切 patch
→ 输出可人工审核的 label overlay
```

因为只有一张图，所以可以先做：

```text
pseudo-label + patch overfit
```

这意味着模型先学会当前这张图上的 Kossel line，先服务当前图像处理，不追求泛化。等后续有更多 tif，再扩展成真正泛化模型。

---

# 3. 第一阶段目录结构

请让 Codex 新建一个项目式输出目录：

```text
line_stage1_dataset/
  previews/
  responses/
  pseudo_labels/
  overlays/
  patches/
    images/
    labels/
    ignore/
  manifests/
  reports/
```

说明：

```text
previews/        保存原图预览、abs 图预览
responses/       保存 Sato / Frangi / Meijering / coherence 等 response
pseudo_labels/   保存 pseudo line mask、ignore mask
overlays/        保存 mask overlay，方便人工审核
patches/         保存训练 patch
manifests/       保存 train_manifest.csv
reports/         保存参数和统计信息
```

---

# 4. 第一阶段 Codex 任务

## 4.1 新建脚本

请新建：

```text
prepare_line_dataset.py
```

运行方式：

```bash
python prepare_line_dataset.py --input data.tif --out line_stage1_dataset
```

如果后续有多张 tif，也要支持：

```bash
python prepare_line_dataset.py --input_dir raw_tifs --out line_stage1_dataset
```

第一版至少必须支持单张 `--input data.tif`。

---

## 4.2 读取图像

要求：

1. 用 `tifffile.imread()` 读取；
2. 转换为 `float32`；
3. 如果 tif 是 3D，默认取第一帧，并打印 warning；
4. 保存原始 shape、min、max、percentile 到 `reports/input_stats.json`。

示例统计：

```json
{
  "filename": "data.tif",
  "shape": [1544, 2048],
  "dtype": "float32",
  "min": -106.0,
  "max": 60.0,
  "p1": -20.0,
  "p50": 0.0,
  "p99": 18.0
}
```

---

## 4.3 保存 preview

保存：

```text
previews/input_preview.png
previews/abs_preview.png
```

显示要求：

```text
input_preview.png 使用原始 signed image 的 p1~p99 显示范围
abs_preview.png 使用 abs(image) 的 0~p99.5 显示范围
```

不要把 tif 改成 8-bit 后再处理。  
8-bit png 只用于查看。

---

## 4.4 计算辅助 response 通道

基于：

```python
abs_img = abs(img)
```

计算以下 response：

```text
response_sato.png
response_frangi.png
response_meijering.png
response_coherence.png
response_gradient.png
response_fused.png
```

建议参数：

```python
sigmas = [2.0, 3.5, 5.5, 8.0]
gaussian pre-smooth sigma = 1.0
structure tensor sigma_d = 2.0
structure tensor sigma_s = 8.0
```

重要要求：

```text
Sato response 要作为重点输出。
之前视觉上 Sato 对 Kossel line 的响应比硬 mask 更好。
```

保存路径：

```text
responses/response_sato.tif
responses/response_sato.png
responses/response_frangi.tif
responses/response_frangi.png
responses/response_meijering.tif
responses/response_meijering.png
responses/response_coherence.tif
responses/response_coherence.png
responses/response_gradient.tif
responses/response_gradient.png
responses/response_fused.tif
responses/response_fused.png
```

---

## 4.5 生成 pseudo line label

第一版不要追求完美，只要生成可用初始标签。

请生成三类区域：

```text
line = 明确 Kossel line
background = 明确背景
ignore = 不确定区域，不参与训练
```

保存为：

```text
pseudo_labels/line_mask_clear.png
pseudo_labels/line_mask_loose.png
pseudo_labels/ignore_mask.png
pseudo_labels/train_label.png
```

其中 `train_label.png` 像素值约定：

```text
0   = background
1   = clear line
255 = ignore
```

建议逻辑：

```python
sato_norm = normalize(response_sato)

clear_line = sato_norm > percentile(sato_norm, 96~98)
loose_line = sato_norm > percentile(sato_norm, 90~94)

ignore = loose_line AND NOT clear_line
background = NOT loose_line
```

然后做形态学约束：

```text
1. 去掉太小的孤立 blob
2. 保留细长结构
3. 对 clear_line 做轻微 dilation，1~2 px
4. 对 ignore 做适度 dilation，保护不确定边界
```

注意：

```text
不要把所有高响应点都当成 line。
Kossel line 交叉点、亮斑、小团状结构可以先进 ignore。
```

---

## 4.6 输出 overlay 供人工检查

必须输出：

```text
overlays/overlay_clear_line.png
overlays/overlay_loose_line.png
overlays/overlay_train_label.png
overlays/overlay_ignore.png
```

显示颜色建议：

```text
clear line: green
ignore: orange/yellow
background: 不画
```

`overlay_train_label.png` 中：

```text
green = clear line
yellow = ignore
```

人工审核重点：

```text
1. green 是否主要覆盖 Kossel line
2. 大量明显背景是否没有被标成 line
3. 弱线和交叉复杂区域是否进 ignore，而不是强行标错
4. 小 compact defect-like 结构是否不要轻易标成 line
```

---

## 4.7 切 patch 生成训练数据

从原图和 response 中切 patch。

默认 patch 参数：

```bash
--patch-size 512
--stride 256
--min-line-ratio 0.002
--max-ignore-ratio 0.7
```

每个 patch 保存：

```text
patches/images/img_000001.npz
patches/labels/label_000001.png
patches/ignore/ignore_000001.png
```

`img_000001.npz` 里保存多通道数组：

```python
channels = {
  "raw": normalized raw signed image,
  "abs": normalized abs image,
  "sato": normalized response_sato,
  "frangi": normalized response_frangi,
  "coherence": normalized response_coherence,
  "gradient": normalized response_gradient
}
```

建议实际保存为：

```python
np.savez_compressed(
    path,
    image=stacked_array,  # shape: [C, H, W] or [H, W, C]
    channel_names=["raw", "abs", "sato", "frangi", "coherence", "gradient"]
)
```

label 保存为 8-bit png：

```text
0   background
1   line
255 ignore
```

---

## 4.8 生成 manifest

保存：

```text
manifests/train_manifest.csv
```

字段：

```text
patch_id,image_path,label_path,ignore_path,x0,y0,w,h,line_ratio,ignore_ratio
```

这样后续训练 U-Net 时可以直接读取。

---

## 4.9 生成 patch contact sheet

为了人工检查 patch 标签质量，输出：

```text
reports/patch_contact_sheet.png
```

内容：

```text
每个 patch 显示 raw preview + label overlay
至少显示前 64 个 line_ratio 较高的 patch
```

如果可以，再输出：

```text
reports/hard_cases_contact_sheet.png
```

用于显示 ignore_ratio 高、交叉线多、标签不确定的 patch。

---

## 4.10 生成 summary

保存：

```text
reports/summary.json
```

内容至少包括：

```json
{
  "num_input_images": 1,
  "patch_size": 512,
  "stride": 256,
  "num_patches": 0,
  "line_pixel_ratio": 0.0,
  "ignore_pixel_ratio": 0.0,
  "response_source": "sato",
  "label_values": {
    "background": 0,
    "line": 1,
    "ignore": 255
  }
}
```

---

# 5. 第一阶段验收标准

运行：

```bash
python prepare_line_dataset.py --input data.tif --out line_stage1_dataset
```

必须生成：

```text
line_stage1_dataset/
  previews/input_preview.png
  previews/abs_preview.png

  responses/response_sato.png
  responses/response_frangi.png
  responses/response_meijering.png
  responses/response_coherence.png
  responses/response_gradient.png
  responses/response_fused.png

  pseudo_labels/line_mask_clear.png
  pseudo_labels/line_mask_loose.png
  pseudo_labels/ignore_mask.png
  pseudo_labels/train_label.png

  overlays/overlay_clear_line.png
  overlays/overlay_loose_line.png
  overlays/overlay_train_label.png
  overlays/overlay_ignore.png

  patches/images/*.npz
  patches/labels/*.png
  patches/ignore/*.png

  manifests/train_manifest.csv

  reports/input_stats.json
  reports/summary.json
  reports/patch_contact_sheet.png
```

我人工优先检查：

```text
overlays/overlay_train_label.png
responses/response_sato.png
reports/patch_contact_sheet.png
```

判断标准：

```text
1. clear line 是否明显覆盖主要 Kossel line；
2. weak/uncertain line 是否进入 ignore；
3. compact 小团状疑似 defect 不要被大量标成 clear line；
4. patch 数量是否足够；
5. patch 标签是否能用于训练 U-Net。
```

---

# 6. 第一阶段不要做的事

本阶段不要做：

```text
1. 不要训练深度学习模型；
2. 不要做背景随机采样；
3. 不要做 line inpainting；
4. 不要做 defect detection；
5. 不要输出 defect candidate；
6. 不要把 pseudo label 说成 ground truth；
7. 不要把所有 Sato 高响应都强行当成 line；
8. 不要把不确定区域强行标为 background。
```

第一阶段只做：

```text
准备 line mask 训练数据。
```

---

# 7. 下一步计划

完成第一阶段后，下一步才是：

```text
Stage 1B:
训练 LineMaskNet / U-Net，让模型从 raw + response channels 学会 line mask。

Stage 2:
基于模型预测的 line mask 做局部随机背景采样 / inpainting。

Stage 3:
contrast comparison 找 compact defect candidate。
```

---

# 8. 给 Codex 的执行提示

你可以直接对 Codex 说：

```text
请阅读 sic_kossel_deep_pipeline_stage1.md。
当前文件夹只有 data.tif。
请新建 prepare_line_dataset.py，按 markdown 中第一阶段任务实现。
先不要训练模型，也不要做 defect detection。
运行命令应为：
python prepare_line_dataset.py --input data.tif --out line_stage1_dataset
```
