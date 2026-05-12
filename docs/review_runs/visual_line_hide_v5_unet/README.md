# Visual Line Hide v5 — with Route B UNet line_prob

## 本轮目标

把 Route B v1 训练出来的 deep U-Net `line_prob.tif` 接入 v5 deepmask
suppression，做第一轮**真正的** deepmask 抑制测试，验证视觉去线效果是否
明显优于：

- **v4 规则版主推**：`v4_rt1.0_cl31_cw1`
- **v5 synth 主推**：`v5_ct0.50_ht0.35_cl0`（合成 prob）

本轮**不做** defect detection。**不再使用** synthetic prob，**全部用真实
UNet line_prob**。

## 关键变化 — line_prob 来源

```
docs/review_runs/route_b_line_mask_v1/  ✅ 已 review 通过
  └─ 训练得到的 line_prob.tif (位于 outputs/route_b_line_mask_v1/)

                ↓ 本轮 v5 通过 --line-prob 直接接入

scripts/route_c_bg_sampling_band_suppress/line_band_contrast_suppress_v5_deepmask.py
```

## 运行命令

```bash
python scripts/route_c_bg_sampling_band_suppress/line_band_contrast_suppress_v5_deepmask.py \
    --input data/data.tif \
    --prepared outputs/prepare \
    --line-prob /e/dapro/dapro-deep-mask-v1/outputs/route_b_line_mask_v1/line_prob.tif \
    --out outputs/visual_line_hide_v5_unet \
    --core-thresholds 0.5 0.6 0.7 \
    --halo-thresholds 0.25 0.35 0.45 \
    --closing-lens 0 15
```

### Sweep 配置

| 维度 | 取值 |
|---|---|
| core_threshold | 0.5 / 0.6 / 0.7 |
| halo_threshold | 0.25 / 0.35 / 0.45 |
| closing_len | 0 / 15 |
| closing_width | 1（固定） |
| halo_dilate | 2（固定） |
| keep_core | 0.0（固定） |
| keep_halo | 0.3（固定） |
| side_sigma | 15（固定） |
| tensor_sigma | 4（固定） |
| distances | {9, 11, 15, 19}（固定） |

3 × 3 × 2 = **18 个变体**，6 分 11 秒跑完（371s）。

## 输入路径

- `data/data.tif`（不进 Git）
- `outputs/prepare/`（Route C 的预处理，跨 worktree 复用）
- `/e/dapro/dapro-deep-mask-v1/outputs/route_b_line_mask_v1/line_prob.tif`
  （Route B 的 UNet 推理结果，跨 worktree 引用）

## 输出路径

### 完整产物（不入 Git）

- `outputs/visual_line_hide_v5_unet/line_prob.tif`、`line_prob_preview.png`、
  `protect_*.png`
- `outputs/visual_line_hide_v5_unet/reports/`
  - `summary.csv`、`contact_sheet.png`、`compare_sheet.png`、
    `zoom_compare_sheet.png`、`baseline_compare.png`、
    `line_prob_info.json`、`zoom_crops.json`
- `outputs/visual_line_hide_v5_unet/variants/<name>/`
  - `summary_panel.png`（含 4 个 zoom crop）
  - `suppressed_preview.png`、`removed.png`、`overlay.png`、`cleanup_mask.png`
  - `core_mask.png`、`halo_mask.png`、`suppressed.tif`、`metrics.json`

### 审核可提交产物（本目录，已入 Git）

- **`baseline_compare.jpg`** — 4 行对照：
  raw / v4 best / v5 synth best / **v5 UNet best**；每行 5 张（全图 + 4 zoom）
- `compare_sheet.jpg` — 18 个变体的 (core+halo / suppressed / removed) 三列
- `contact_sheet.jpg` — 18 个 suppressed 缩略
- `zoom_compare_sheet.jpg` — Top 5 + raw 行 × 4 zoom
- `line_prob_preview.jpg` — UNet 出的 line_prob 全图视觉
- `protect_true_blob.jpg` / `protect_on_line.jpg` (同 v4/v5_synth)
- `summary.csv`、`shortlist.md`、`README.md`
- `variants/<top>/{summary_panel,suppressed_preview,removed}.jpg` × 7

所有 JPG 最长边 ≤ 2800 px，quality=85。

## 重点查看文件（按优先级）

1. **`baseline_compare.jpg`** —— 本轮**最关键的一张**。直接对比 raw vs v4 vs
   v5 synth vs v5 UNet 在 4 个 zoom 区域的视觉效果。
2. **`zoom_compare_sheet.jpg`** —— v5 UNet 5 个 top 变体 vs raw 在 4 个 zoom 上。
3. `line_prob_preview.jpg` —— UNet 出的 line_prob 全图是否合理（连续、覆盖
   黑线、背景干净）。
4. `shortlist.md` —— 7 个候选 + 选型说明 + v3/v4/v5 主推对比表。
5. `variants/v5_ct0.50_ht0.45_cl0/summary_panel.jpg` —— 主推变体细节。
6. `summary.csv` 中重点列：
   - `core_coverage_percent`：22-32%（vs v5 synth 1.58%，**14× 宽**）
   - `edit_coverage_percent`：33-39%
   - `dir_bg_coverage_core_percent`：62-69%（方向性 bg 命中率比 v3/v4 高）
   - `core_residual_after_suppress`：-88 ~ -104（**负数**，见下解释）

## 关于 `core_residual` 整体为负

v5 UNet 所有变体的 `core_resid` 都在 -89 ~ -104。这**不**是病：

- `core_resid = bg_low[core] - suppressed[core]`，其中 `bg_low = gauss(raw, σ=20)`。
- 当 core 覆盖率从 8%（v3/v4）→ 22%（v5 UNet）时，**bg_low 本身**在线位置
  被线的暗度严重污染（σ=20 的 Gaussian 把 22% 的暗像素卷入平均）。
- 替换后的 `suppressed[core]` ≈ 真背景值（用线带外的方向性采样得到），
  比受污染的 `bg_low[core]` 亮，所以差值变成负数。
- **这是 metric 本身在高 core_cov 时失效**，不是 over-shoot。

判断 v5 UNet 是否过度修正必须看视觉（zoom_compare 里有没有出现亮带 /
新灰带），不能再用 core_resid。

## 重点判断（review checklist）

按用户给的 3 个重点：

1. **长贯穿黑线是否更完整地被压下去？**
   → 看 baseline_compare 第 4 行（v5 UNet）vs 第 2 行（v4） vs 第 3 行（v5 synth）
   的"full" 全图列与 Z1/Z3 长线 zoom。
2. **中心星芒区域是否比 v4 / v5 synth 更干净？**
   → 看 zoom Z2 / Z3（中心交叉点）的对比。
3. **是否出现新的灰带或大面积误修？**
   → 看 v5 UNet 行的全图：是否在原本不是线的位置出现整片灰色或亮色斑块。
   → 风险变体：#5 (closing)、#6/#7 (最宽 halo)。

## Top variants

| # | variant | core% | edit% | 备注 |
|---|---|---:|---:|---|
| 1 | `v5_ct0.50_ht0.45_cl0`  | 22.59 | 33.15 | **主推** |
| 2 | `v5_ct0.60_ht0.45_cl0`  | 22.02 | 33.15 | 稍严格 core |
| 3 | `v5_ct0.70_ht0.45_cl0`  | 21.42 | 33.15 | 最严格 core |
| 4 | `v5_ct0.50_ht0.35_cl0`  | 22.59 | 33.93 | 更宽 halo |
| 5 | `v5_ct0.50_ht0.35_cl15` | 32.03 | 38.24 | +closing |
| 6 | `v5_ct0.50_ht0.25_cl0`  | 22.59 | 34.91 | 最宽 halo |
| 7 | `v5_ct0.50_ht0.25_cl15` | 32.03 | 38.97 | **最激进**（灰带风险） |

详见 `shortlist.md`。

## 本轮明确不做

- ❌ 不做 defect detection
- ❌ 不使用 v5 synthetic prob
- ❌ 不修改 Route B 的训练 / 推理脚本（接口已 ready，直接用 line_prob.tif）
- ❌ 不修改 `main` 分支；所有改动只在 worktree `experiment/visual-line-hide-v1`
- ❌ 不提交 `data/`、`outputs/`、`.tif/.npz/.npy/.pt` 等大文件

v5 UNet 本轮全部任务是 **接入 Route B 真 line_prob + 验证 deepmask 是否
显著优于 v4 / v5_synth**。
