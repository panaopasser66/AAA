# Task: Stage 1B - Train LineMaskNet v1 from Selected Pseudo-label

## 0. 当前状态

当前已经完成 Stage 1A：

```text
data.tif
line_stage1_dataset/
  pseudo_labels/train_label_selected.png
  patches/images/*.npz
  patches/labels/*.png
  patches/ignore/*.png
  manifests/train_manifest.csv
  reports/patch_contact_sheet.png
  reports/hard_cases_contact_sheet.png
  reports/summary.json
```

选用的 pseudo-label 是：

```text
D_clear98_loose80
clear = sato > p98
loose = sato > p80
line pixel ratio = 2.79%
ignore pixel ratio = 28.41%
patch count = 81
patch size = 512
stride = 256
```

现在进入 Stage 1B：

```text
训练第一版 LineMaskNet / U-Net，让模型学习 Kossel line mask。
```

本阶段目标不是 defect detection，也不是去线修复。  
本阶段只做：

```text
训练 line segmentation model
→ 在整张 data.tif 上推理
→ 输出 line probability map / hard mask / overlay
→ 判断模型是否比 pseudo-label 更自然、更连续
```

---

# 1. 本阶段不要做的事

不要做：

```text
1. 不要做背景随机采样；
2. 不要做 inpainting；
3. 不要做 defect detection；
4. 不要做 contrast comparison；
5. 不要把模型预测结果直接当最终 defect 结果；
6. 不要修改原始 data.tif；
7. 不要删除 Stage 1A 输出。
```

---

# 2. 需要新建的脚本

请新建两个脚本：

```text
train_line_mask_unet.py
infer_line_mask_unet.py
```

也可以把 inference 放在训练脚本中，但推荐分开。

---

# 3. 训练脚本要求

运行方式：

```bash
python train_line_mask_unet.py --dataset line_stage1_dataset --out line_stage1_train_v1 --epochs 120 --batch-size 4 --lr 1e-3
```

需要读取：

```text
line_stage1_dataset/manifests/train_manifest.csv
line_stage1_dataset/patches/images/*.npz
line_stage1_dataset/patches/labels/*.png
```

label 约定：

```text
0   = background
1   = clear Kossel line
255 = ignore
```

训练 loss 中必须忽略 `label == 255` 的像素。

---

# 4. 输入通道

每个 `.npz` 中可能保存：

```text
raw
abs
sato
frangi
coherence
gradient
```

或者保存为一个多通道 `image` 数组。

请自动兼容以下情况：

```python
data = np.load(npz_path)
if "image" in data:
    image = data["image"]
else:
    stack available named channels
```

最终输入 tensor shape 应为：

```text
[C, H, W]
```

如果 `.npz` 是 `[H, W, C]`，需要自动转为 `[C, H, W]`。

保存训练使用的 channel_names 到：

```text
line_stage1_train_v1/config.json
```

---

# 5. 模型结构

第一版用轻量 U-Net 即可。

要求：

```text
1. 输入通道数自动根据 npz 决定；
2. 输出 1 个 logit channel；
3. 输出经过 sigmoid 后是 line probability；
4. 模型不要太大，适合单张图 patch overfit。
```

推荐结构：

```text
Encoder: 32 → 64 → 128 → 256
Decoder: 256 → 128 → 64 → 32
skip connections
BatchNorm or GroupNorm
ReLU or SiLU
```

如果 GPU 不可用，CPU 也要能跑，只是慢一点。

---

# 6. Loss 设计

由于 line pixel ratio 只有约 2.79%，类别极不平衡，不能只用普通 BCE。

请实现：

```text
masked BCEWithLogitsLoss + masked Dice loss
```

其中：

```text
ignore pixels 不参与 BCE，也不参与 Dice。
```

建议：

```python
loss = bce_loss + 0.5 * dice_loss
```

`pos_weight` 可以从训练集自动估计：

```text
pos_weight = num_background_pixels / num_line_pixels
```

为了稳定，建议 clamp：

```text
pos_weight clamp 到 [3, 30]
```

---

# 7. 数据划分

当前只有 81 个 patch，而且都来自同一张图，所以 validation 不能代表真实泛化，只作为监控。

做法：

```text
train : val = 85 : 15
```

要求：

```text
固定 random seed = 42
```

保存：

```text
line_stage1_train_v1/splits/train_ids.txt
line_stage1_train_v1/splits/val_ids.txt
```

---

# 8. 数据增强

第一版只做安全增强，不要做会破坏线结构的强增强。

建议训练时随机：

```text
1. horizontal flip
2. vertical flip
3. 90-degree rotation
4. small Gaussian noise
5. mild intensity scale/shift
```

不要做：

```text
1. 大角度旋转导致插值变形；
2. 强 elastic deformation；
3. 强 blur；
4. CutMix/MixUp。
```

---

# 9. 训练输出

训练输出目录：

```text
line_stage1_train_v1/
  checkpoints/
    best.pt
    last.pt
  previews/
  logs/
  curves/
  config.json
  metrics.csv
```

每个 epoch 记录：

```text
epoch
train_loss
val_loss
val_bce
val_dice_loss
val_precision
val_recall
val_f1
val_iou
learning_rate
```

保存到：

```text
line_stage1_train_v1/metrics.csv
```

画训练曲线：

```text
line_stage1_train_v1/curves/loss_curve.png
line_stage1_train_v1/curves/metric_curve.png
```

---

# 10. 训练过程中的 prediction preview

每隔 10 个 epoch，或者训练结束时，对固定 val patch 输出预测图：

```text
line_stage1_train_v1/previews/epoch_010_val_predictions.png
line_stage1_train_v1/previews/epoch_020_val_predictions.png
...
line_stage1_train_v1/previews/final_val_predictions.png
```

每个 preview 至少展示：

```text
raw patch
selected label overlay
predicted probability
predicted hard mask overlay
```

颜色：

```text
green = predicted line
yellow = ignore region
red = false positive / optional
blue = false negative / optional
```

---

# 11. 推理脚本要求

运行方式：

```bash
python infer_line_mask_unet.py --model line_stage1_train_v1/checkpoints/best.pt --dataset line_stage1_dataset --input data.tif --out line_stage1_infer_v1
```

推理脚本需要：

```text
1. 读取训练 config.json，知道 channel_names；
2. 对整张 data.tif 计算或读取与训练一致的 response channels；
3. 使用 sliding window 推理；
4. patch size 默认 512；
5. stride 默认 256；
6. overlap 区域用 averaging 融合 probability；
7. 输出整图 line probability map。
```

如果已有：

```text
line_stage1_dataset/responses/*.tif
```

优先读取已有 response，避免重复计算。

---

# 12. 推理输出

输出目录：

```text
line_stage1_infer_v1/
  line_prob.tif
  line_prob.png

  line_mask_thr03.png
  line_mask_thr05.png
  line_mask_thr07.png

  overlay_thr03.png
  overlay_thr05.png
  overlay_thr07.png

  compare_selected_label_vs_prediction.png
  inference_summary.json
```

其中：

```text
line_prob.tif = float32 probability map
line_prob.png = 0~1 显示
line_mask_thrXX = probability > threshold
overlay_thrXX = mask overlay on input preview
```

---

# 13. 推理结果 contact sheet

生成：

```text
line_stage1_infer_v1/inference_contact_sheet.png
```

至少包含：

```text
1. input_preview
2. response_sato
3. selected_label_overlay
4. line_prob
5. overlay_thr03
6. overlay_thr05
7. overlay_thr07
8. difference between selected label and prediction
```

---

# 14. 我会怎么判断结果

训练完成后，我会重点看：

```text
line_stage1_infer_v1/inference_contact_sheet.png
line_stage1_infer_v1/overlay_thr03.png
line_stage1_infer_v1/overlay_thr05.png
line_stage1_infer_v1/line_prob.png
line_stage1_train_v1/previews/final_val_predictions.png
```

判断标准：

```text
1. line_prob 是否沿 Kossel line 连续，而不是断成碎片；
2. 强 Kossel line 是否高概率；
3. 弱 Kossel line 是否至少有中等概率；
4. 背景是否低概率；
5. compact 小团状结构是否不要全部被高概率覆盖；
6. 交叉点可以高概率，但不要糊成大片；
7. thr03 / thr05 / thr07 三档中至少有一档适合作为后续背景修复 mask。
```

---

# 15. 如果训练过拟合怎么办

当前只有一张图和 81 个 patch，过拟合是可以接受的，因为这是 v1 proof-of-concept。

本阶段目标是：

```text
让模型学会当前图里的 Kossel line pattern。
```

不是：

```text
训练一个泛化到所有样品的最终模型。
```

如果模型预测比 pseudo-label 更平滑、更连续，这是好事。  
如果模型只是完全复制 pseudo-label 且弱线仍漏检，下一轮再调整 label 或加入更多图。

---

# 16. 验收命令

请最终能运行：

```bash
python train_line_mask_unet.py --dataset line_stage1_dataset --out line_stage1_train_v1 --epochs 120 --batch-size 4 --lr 1e-3
```

然后：

```bash
python infer_line_mask_unet.py --model line_stage1_train_v1/checkpoints/best.pt --dataset line_stage1_dataset --input data.tif --out line_stage1_infer_v1
```

---

# 17. 最后打印

训练脚本结束时打印：

```text
Training done.
Please check:
  line_stage1_train_v1/curves/loss_curve.png
  line_stage1_train_v1/previews/final_val_predictions.png
  line_stage1_train_v1/checkpoints/best.pt
```

推理脚本结束时打印：

```text
Inference done.
Please check:
  line_stage1_infer_v1/inference_contact_sheet.png
  line_stage1_infer_v1/overlay_thr03.png
  line_stage1_infer_v1/overlay_thr05.png
  line_stage1_infer_v1/overlay_thr07.png
```
