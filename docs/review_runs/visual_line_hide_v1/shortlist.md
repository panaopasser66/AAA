# Visual Line Hide v1 — Top Variant Shortlist

本表来自 108 个 Route C 变体的 `summary.csv` 初筛。

挑选原则：
1. **脚本作者预定的 `PRIORITY_NAMES`**（`side_band` + `sigma=15` + `keep=0.1` 主轴，
   覆盖 B/C/D 三档 mask、宽度 4 与 6）
2. 补充 2 个 `masked_gaussian` 高 `removed_inside_outside_ratio` 的对照
   （B_sato95 + sigma15 系列）

“ratio” 越大代表 contrast 变化越集中在线带内部，外部几乎不动；ratio 高 + cov 适
中、`mean_abs_removed_inside` 大的变体应优先肉眼复核。

| # | variant_name | mask | width | method | sigma | keep | cov% | in | out | ratio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | B_sato95_width6_mg_sigma15_keep0.1 | B_sato95 | 6 | masked_gaussian | 15 | 0.1 | 18.83 | 212.8 | 2.13 | **100.12** |
| 2 | B_sato95_width4_mg_sigma15_keep0.1 | B_sato95 | 4 | masked_gaussian | 15 | 0.1 | 14.91 | 217.6 | 2.40 | 90.47 |
| 3 | B_sato95_width4_side_sigma15_keep0.1 | B_sato95 | 4 | side_band | 15 | 0.1 | 14.91 | 218.3 | 2.88 | 75.76 |
| 4 | C_sato90_width6_side_sigma15_keep0.1 | C_sato90 | 6 | side_band | 15 | 0.1 | 32.88 | 203.0 | 3.99 | 50.93 |
| 5 | C_sato90_width4_side_sigma15_keep0.1 | C_sato90 | 4 | side_band | 15 | 0.1 | 26.62 | 205.0 | 4.15 | 49.40 |
| 6 | D_sato85_width6_side_sigma15_keep0.1 | D_sato85 | 6 | side_band | 15 | 0.1 | 45.55 | 193.6 | 5.61 | 34.50 |
| 7 | D_sato85_width4_side_sigma15_keep0.1 | D_sato85 | 4 | side_band | 15 | 0.1 | 37.39 | 194.8 | 5.67 | 34.35 |

列含义：
- `cov%` = `edit_coverage_percent`（编辑区占整图比例）
- `in` = `mean_abs_removed_inside`（线带内部的平均 `|raw - suppressed|`）
- `out` = `mean_abs_removed_outside`（线带外部）
- `ratio` = `in / out`

## 选哪一个

- **追求最“干净”地只动线、且线本身较窄时**：#1 / #2（B_sato95 + masked_gaussian）。
  cov 小、外部几乎零扰动；但 mask 较严，可能漏掉粗线。
- **平衡覆盖 + 隐藏强度**：#3 / #4 / #5（B/C + side_band）。
  side_band 用线带外的环带采样，处理边界更稳。
- **激进消去所有 sato85 以上能找到的线**：#6 / #7（D_sato85）。cov 高（37–45%），
  会动到一些灰带，要小心是否压扁了背景结构；ratio 也较低。

下一步建议：在 `variants/<name>/suppressed_preview.jpg` 和 `removed.jpg` 上逐一肉
眼比对，最终选 1–2 个 variant 作为「视觉隐线」的 hand-picked 输出。

## 对应图位

每个 variant 在本目录下都有：

- `variants/<name>/summary_panel.jpg` — raw / edit_mask / suppressed / diff 四联图
- `variants/<name>/suppressed_preview.jpg` — 抑制后单图全分辨率
- `variants/<name>/removed.jpg` — `raw - suppressed` 的红蓝热图
