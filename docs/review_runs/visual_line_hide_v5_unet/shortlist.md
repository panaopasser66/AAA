# Visual Line Hide v5 (UNet line_prob) — Top Variant Shortlist

来自 18 个 v5+真 line_prob 变体的初筛。本轮 line_prob 来自 Route B v1
（**真正的 deep U-Net 推理结果**，不是 v5 之前的合成 prob）。

## line_prob 上下文

| 指标 | v5 synth | **v5 UNet** |
|---|---:|---:|
| prob p50 | 0.142 | **0.001** |
| prob p90 | 0.258 | **1.000** |
| prob p99 | 0.567 | **1.000** |
| prob > 0.5 cov | 1.58% | **22.61%** |
| 性质 | 偏 unimodal 中等值 | bimodal sharp (0/1) |

## v5 UNet 18 个变体（按 closing_len 分组）

`core_th × halo_th × closing_len` 共 3 × 3 × 2 = 18。

> 所有变体的 `mean_abs_removed_outside = 0.00`（编辑严格限定在 core+halo 内）。
> `dir_bg_coverage_core ≈ 62-69%`（方向性 bg 在 core 上的成功率比 v3/v4 高）。

| variant | core% | edit% | dir% | rm_core | rm_halo | core_resid |
|---|---:|---:|---:|---:|---:|---:|
| `v5_ct0.50_ht0.45_cl0`  | 22.59 | **33.15** | **69.2** | 307.6 |  97.1 | **-88.73** |
| `v5_ct0.60_ht0.45_cl0`  | 22.02 | 33.15 | 69.0 | 311.3 |  98.0 | -89.11 |
| `v5_ct0.70_ht0.45_cl0`  | 21.42 | 33.15 | 68.8 | 315.5 |  98.8 | -89.49 |
| `v5_ct0.50_ht0.35_cl0`  | 22.59 | 33.93 | 68.8 | 308.8 |  98.0 | -90.32 |
| `v5_ct0.50_ht0.35_cl15` | 32.03 | 38.24 | 63.3 | 264.7 |  98.6 | -101.75 |
| `v5_ct0.50_ht0.25_cl0`  | 22.59 | **34.91** | 68.3 | 310.3 |  99.1 | -92.28 |
| `v5_ct0.50_ht0.25_cl15` | 32.03 | **38.97** | 62.9 | 265.6 | 100.2 | **-103.28** |

(完整 18 行见 `summary.csv`)

## 关于 `core_residual` 为负

v5_UNet 的所有变体 `core_residual ≈ -89 ~ -104`（负数），意思是 **suppressed 比
bg_low 偏亮**。

- v3 / v4 (规则 mask, core_cov ≈ 8%)：core_resid 是 **正数 +18 ~ +25**——
  suppressed 还比 bg_low 偏暗（线没压到位）
- v5 synth (synth prob, core_cov 1.58%)：core_resid ≈ **-0.6**——刚好压平
- **v5 UNet (UNet prob, core_cov 22%)：core_resid ≈ -90**——过度修正

物理解释：
- `bg_low = gaussian(raw, σ=20)`，σ=20 的 Gaussian 把约 20% 的图像面积
  纳入平均；线占了 ~22% 的像素，所以 bg_low **本身已经被线的暗度污染**，
  在线位置 bg_low 介于"线值"和"真正背景值"之间。
- 替换后 `suppressed[core] = bg_dir[core] ≈ 真正背景值`（bg_dir 用了线带外
  的两侧采样，不含线污染）。
- `bg_low[core] - suppressed[core] = 中等值 - 真背景值 < 0`，因此负数。

**所以 core_resid 在 v5 UNet 下不是"过度修正"，而是 bg_low 这个度量本身在
高 core_cov 时不再公平**。判断 v5 UNet 的好坏要看视觉（`zoom_compare_sheet`、
`baseline_compare`），不能再看 core_resid。

## 7 个候选

| # | variant_name | core% | edit% | 备注 |
|---|---|---:|---:|---|
| 1 | `v5_ct0.50_ht0.45_cl0`  | 22.59 | 33.15 | **主推**：core_th=0.5 + 最紧 halo + 无 closing |
| 2 | `v5_ct0.60_ht0.45_cl0`  | 22.02 | 33.15 | 稍严格 core |
| 3 | `v5_ct0.70_ht0.45_cl0`  | 21.42 | 33.15 | 最严格 core |
| 4 | `v5_ct0.50_ht0.35_cl0`  | 22.59 | 33.93 | 更宽 halo (0.35)，无 closing |
| 5 | `v5_ct0.50_ht0.35_cl15` | 32.03 | 38.24 | +closing：core 增到 32% (沿线断点缝合) |
| 6 | `v5_ct0.50_ht0.25_cl0`  | 22.59 | 34.91 | 最宽 halo (0.25)，无 closing |
| 7 | `v5_ct0.50_ht0.25_cl15` | 32.03 | 38.97 | **最激进**：closing + 最宽 halo（灰带风险） |

## 怎么挑

- **默认主推 #1** (`v5_ct0.50_ht0.45_cl0`) ——
  edit_cov 最小、core_resid 量级最小、dir bg 在 core 上覆盖率最高（69.2%）。
- 看 `baseline_compare.jpg` 同时对比 raw / v4 best / v5 synth best / v5 UNet best。
- 看 `zoom_compare_sheet.jpg` 验证：
  - 中心星芒区域 (Z2 ~ Z3) 是否比 v4/v5 synth 更干净
  - 长贯穿黑线是否真的完整消失
  - 有没有新出现的"灰带"或"亮带"（v5_UNet 的 over-correction 风险）
- 如果担心残留：试 #5 (closing) 把沿线断点也封掉。
- 如果担心灰带：避开 #5 / #7（closing + 宽 halo）。

## v3 / v4 / v5_synth / v5_UNet 主推对比

| | v3 | v4 | v5 synth | **v5 UNet** |
|---|---:|---:|---:|---:|
| 变体 | `C90b_bt1.0_wc7_h6_kh0.3` | `v4_rt1.0_cl31_cw1` | `v5_ct0.50_ht0.35_cl0` | `v5_ct0.50_ht0.45_cl0` |
| Mask 来源 | sato+black 规则 | + cleanup pass | synth prob | **UNet prob** |
| core_cov% | 8.02 | 8.02 | 1.58 | **22.59** |
| edit_cov% | 18.55 | varies | 14.07 | 33.15 |
| dir bg cov on core% | 59.3 | 56.7 | 69.7 | 69.2 |
| rm_core (mean \|diff\|) | 260.6 | 254.9 | n/a | 307.6 |
