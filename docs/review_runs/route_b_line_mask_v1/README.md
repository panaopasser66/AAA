# Route B — Line Mask v1 (deep line probability)

## 本轮目标

为 Route C v5 deepmask suppression 提供一张更连续、更宽、更像人眼判断的
Kossel 线概率图 `line_prob.tif`。

> v5 review 结论：合成 prob 太窄（p99=0.57，prob>0.5 只覆盖 1.58%）导致 v5
> 主推 core_cov 只有 1.58%（vs v4 的 8.02%），断续黑点残留明显。

本轮**不做** defect detection、**不做** line suppression。**只输出** line_prob。

## 关键流水线

```
data/data.tif
  │
  ├─ scripts/route_b_line_mask_unet/prepare_line_dataset_v2.py
  │   └─ outputs/route_b_prepare_v2/    (responses + pseudo labels + 81 patches)
  │
  ├─ scripts/route_b_line_mask_unet/train_line_mask_unet.py
  │   └─ outputs/route_b_train_v1/      (40 epoch overfit, best epoch=33)
  │
  └─ scripts/route_b_line_mask_unet/infer_line_mask_unet_v2.py
      └─ outputs/route_b_line_mask_v1/  (line_prob.tif + overlays)
```

## 一个重要 bug 修复

v1 的 `prepare_line_dataset.py` 用了

```python
sato(abs_img, black_ridges=False)
```

—— 这是 **bright-ridge** 极性。对于 Kossel 线（**dark ridges** in raw），
sato 实际上检测的是线**之间**的亮带，不是线本身。实测：

| | sato(bright_ridges) | sato(dark_ridges, raw) |
|---|---:|---:|
| 在 top 5% 最暗像素上的均值 | 21 | 197 |
| 在自己 top 5% 像素的 raw 上 | bg+0 | bg-620 |
| dark_residual 在自己 top 5% 上的均值 | -203 (lighter than bg) | +440 (darker) |

> **解读**：v1 的 sato 实际上和"暗线"几乎无关。这就是为什么 v3/v5 里
> `sato_mask ∩ (black_score > 1.0σ)` 在 v1 sato 下覆盖率只有 0.15%。

v2 改用 `sato(raw, sigmas=..., black_ridges=True)`：在同样的 sato 阈值（p87）+
black_score (>1.0σ) AND 下，positive coverage 立刻从 ~0% 涨到 **12.6%**。

详见 `prepare_line_dataset_v2.py` 顶部 docstring。

## 运行命令

```bash
# 1) prepare: 响应通道 + 黑线感知伪标签 + 81 个训练 patch
python scripts/route_b_line_mask_unet/prepare_line_dataset_v2.py \
    --input data/data.tif \
    --out outputs/route_b_prepare_v2

# 2) 训练: 1.93M 参数 U-Net, batch=2, lr=1e-3, 40 epoch
python scripts/route_b_line_mask_unet/train_line_mask_unet.py \
    --dataset outputs/route_b_prepare_v2 \
    --out outputs/route_b_train_v1 \
    --epochs 40 --batch-size 2 --base-channels 32 --num-workers 0 \
    --preview-every 10 --lr 1e-3

# 3) 全图推理 + overlays + contact sheet
python scripts/route_b_line_mask_unet/infer_line_mask_unet_v2.py \
    --model outputs/route_b_train_v1/checkpoints/best.pt \
    --dataset outputs/route_b_prepare_v2 \
    --input data/data.tif \
    --out outputs/route_b_line_mask_v1
```

## 输入通道（6）

`raw_norm`, `abs_norm`, **`sato(raw, black_ridges=True)`**, **`black_line_score`**,
`structure_tensor_coherence`, `gradient_magnitude`。

`abs_norm` 保留是为了沿用现有 train.py 的 channel 数；对于本图（全正值的 raw），
abs 和 raw 等价，模型可忽略。

## Pseudo-label 策略

```
positive (1):  sato_norm > p87  AND  black_line_score > 1.0σ
               -> 12.57% 覆盖 (足够覆盖黑线宽度，比 v1 的 ~3% 宽 4×)

ignore  (255): (sato_norm > p75 OR black > 0.5σ) \ positive
               + 围绕 positive 的 4 px buffer
               -> 32.55% 覆盖 (弱线/边界/交叉点全部进 ignore)

background (0): 其余 54.88%  (清晰非线区)
```

**关键原则**：visible 但弱的 Kossel 线 → 进 ignore；**绝不**作为负样本。

## 训练摘要

- U-Net 1.93M 参数（base=32，4 levels: 32/64/128/256）
- 输入 6 channels, patch 512×512, batch=2
- AdamW lr=1e-3 cosine annealing 到 1e-5
- masked BCE + 0.5 × masked Dice，pos_weight=4.42
- 40 epoch ≈ 5 分钟（NVIDIA MX450）
- val_loss: epoch1 0.262 → epoch10 0.010 → epoch40 0.0018
- best val_F1 = **0.9999 @ epoch 33** (单图允许 overfit，符合任务要求)

详见 `loss_curve.jpg`, `metric_curve.jpg`, `final_val_predictions.jpg`,
`training_metrics.csv`。

## 推理摘要

`line_prob.tif`：float32, shape 2450 × 2340, 范围 [1.2e-5, 1.0]。

| 阈值 | 覆盖率 | 用途 |
|---|---:|---|
| prob > 0.3 | **23.85%** | 宽 halo（用作 Route C v5 halo_th=0.25-0.35 的参考） |
| prob > 0.5 | **22.61%** | 中等核心（v5 core_th=0.5 的参考；比 synth prob 的 1.58% 宽 14×） |
| prob > 0.7 | **21.43%** | 高置信核心（v5 core_th=0.7 的参考） |

分布：`p50=0.001, p90=1.000, p99=1.000` —— bimodal（背景几乎为 0，线上几乎
为 1），过渡带很窄（0.3-0.7 之间只 ~2.4% 像素）。

## 重点查看文件（review）

1. **`line_prob_preview.jpg`** —— 全图 prob 灰度。
   - 看主黑线是否被亮色覆盖（连续 + 宽度）
   - 看背景是否暗（无大片误报）
2. **`overlay_thr05.jpg`** —— 主推阈值的 mask 绿色叠加。
   - 是否覆盖了肉眼可见的所有黑线？
   - 有没有把非线区域错标？
3. **`overlay_thr03.jpg`** —— 更宽阈值；用于 halo 选型。
4. **`overlay_thr07.jpg`** —— 更紧阈值；用于核心选型。
5. **`inference_contact_sheet.jpg`** —— 一张图看全流程：
   raw / sato (dark) / black_score / train_label overlay / prob / thr 三档 /
   diff (TP/FP/FN)。
6. **`label_vs_prediction.jpg`** —— prediction vs pseudo-label 直接对比
   （diff: TP 绿 / FP 红 / FN 蓝）。
7. **`prepare_train_label_overlay.jpg`** —— 检查训练 label 本身是否合理
   （绿=positive, 黄=ignore）。
8. **`response_sato_preview.jpg`** + **`response_black_score_preview.jpg`**
   —— 这两个是 label 构造的核心输入，应在黑线处亮。

## 输出路径

### 完整产物（不入 Git）

- `outputs/route_b_prepare_v2/` — 响应 tif、pseudo labels、patches、manifests
- `outputs/route_b_train_v1/` — checkpoints (best/last.pt)、curves、previews
- `outputs/route_b_line_mask_v1/` — **`line_prob.tif`**、line_mask_thr0X.png、
  overlay_thr0X.png、inference_contact_sheet.png

### 审核可提交产物（本目录，已入 Git）

- `README.md`、`metrics.json`、`prepare_summary.json`、`inference_summary.json`、
  `training_metrics.csv`
- `line_prob_preview.jpg`、`overlay_thr03/05/07.jpg`、
  `inference_contact_sheet.jpg`、`label_vs_prediction.jpg`
- `loss_curve.jpg`、`metric_curve.jpg`、`final_val_predictions.jpg`
- `prepare_train_label_overlay.jpg`、`prepare_positive_overlay.jpg`、
  `prepare_ignore_overlay.jpg`
- `response_sato_preview.jpg`、`response_black_score_preview.jpg`
- `patch_contact_sheet.jpg`

所有 JPG 最长边 ≤ 2800 px，quality=85。

## v5 vs Route B v1 line_prob

| | v5 synth prob | Route B v1 line_prob |
|---|---:|---:|
| p50 | 0.142 | 0.001 |
| p90 | 0.258 | 1.000 |
| p99 | 0.567 | 1.000 |
| 覆盖率 @ 0.5 | 1.58% | **22.61%** |
| 覆盖率 @ 0.3 | 9.93% | 23.85% |
| 性质 | 偏 unimodal 中等值 | bimodal (0/1 sharp) |

**意义**：把它接入 Route C v5 (`--line-prob outputs/route_b_line_mask_v1/line_prob.tif`)
预期：v5 主推变体的 core_cov 会从 1.58% 涨到 ~22%，与 v3/v4 的 ~8% 相当或更宽，
方向性 bg 的样本应当更充分。

## 本轮明确不做

- ❌ 不做 defect detection
- ❌ 不做 line suppression（不修改 Route C）
- ❌ 不验证 v5 + line_prob 的最终结果（等 review 通过后再接）
- ❌ 不修改 main 分支；所有改动只在 worktree `experiment/deep-line-mask-v1`
- ❌ 不提交 `data/`、`outputs/`、`*.tif/*.npz/*.npy/*.pt` 等大文件

下一步（待 review 决定）：把 `outputs/route_b_line_mask_v1/line_prob.tif`
拷给 visual-line-hide v5 worktree 接入。
