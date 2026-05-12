# Visual Line Hide v4 — Top Variant Shortlist

来自 15 个 v4 变体的 `summary.csv` 初筛。所有变体共享同一个 **v3 baseline**
(`C90b_bt1.0_wc7_h6_kh0.3`)，只是 cleanup 三参数不同。

## v3 baseline 上下文（所有变体共享）

| | |
|---|---:|
| `core_coverage` | 8.02% |
| `edit_coverage` | 18.55% |
| `residual_p99_v3` | **2.65** |
| `core_residual_v3` (相对 v4 重算的 bg_low_after) | +24.00 |

## v4 cleanup 关键指标

- `cleanup_cov%`：cleanup mask 覆盖率
- `on_line`：cleanup pixels 中**直接落在 broad_line_mask 内**的比例。越高 = 越
  "就线论线"，越低 = 越倾向修 halo 周边
- `dir%`：cleanup 像素中通过方向性背景采样成功的比例
- `rm_avg`：cleanup 区域平均 `|v3_suppressed - v4_suppressed|`，越大 = cleanup
  动作越激进
- `p99_v4`：cleanup 后全图 residual_dark_score 的 99 分位，越低越好
- `core_v4`：cleanup 后核心区残留 darkness（在新 bg_low 下度量）

## 7 个候选

| # | variant_name | rt | cl | cw | cleanup_cov% | on_line | dir% | rm_avg | p99_v4 | core_v4 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `v4_rt1.0_cl31_cw1` | 1.0 | 31 | 1 |  9.39 | **0.56** | 56.7 | 254.9 | 2.58 | **+18.7** | 主推：最高 on_line + 最低 core_v4 |
| 2 | `v4_rt1.0_cl15_cw1` | 1.0 | 15 | 1 |  6.87 | 0.40 | 57.2 | 301.8 | **2.56** | +21.3 | 全图 p99 最低 |
| 3 | `v4_rt1.0_cl0`      | 1.0 |  0 | – |  5.57 | 0.26 | 64.6 | 351.8 | 2.56 | +19.7 | 最保守，cleanup_cov 最小 |
| 4 | `v4_rt0.8_cl31_cw1` | 0.8 | 31 | 1 | 11.18 | 0.51 | 57.3 | 252.1 | 2.61 | +19.6 | 中等阈值 + 强 closing |
| 5 | `v4_rt0.8_cl15_cw1` | 0.8 | 15 | 1 |  8.74 | 0.38 | 58.6 | 289.0 | 2.60 | +21.4 | 中等阈值 + 中等 closing |
| 6 | `v4_rt0.5_cl31_cw1` | 0.5 | 31 | 1 | 14.91 | 0.46 | 58.6 | 240.5 | 2.67 | +20.9 | **激进**，可能出现 v3 那种刷痕 |
| 7 | `v4_rt0.5_cl0`      | 0.5 |  0 | – | 11.45 | 0.29 | 64.3 | 282.2 | 2.66 | +19.7 | **最激进** 且 on_line 最低，最容易乱修 |

(rt = residual_threshold, cl = closing_len, cw = closing_width)

## 主要观察

1. **closing_width 几乎不影响结果**：`cw=1` vs `cw=2` 的指标差异 <0.06%。
   v4 sweep 中仅保留 `cw=1` 作为代表（`cw=2` 的对应变体在 `outputs/...` 里
   仍可查，但 review 包里没收录）。

2. **`closing_len` 提升 on_line 比例**：从 0 → 31 让 on_line 从 ~0.27 提到
   ~0.56。这是 directional closing 沿线方向填补 v3 core 断点的直接效果。

3. **`rt=1.0` 比 `rt=0.5` 更保险**：
   - `rt=0.5` cleanup_cov ≈ 11-15%，on_line 0.29-0.46 — 修了不少 halo 灰带
   - `rt=1.0` cleanup_cov ≈ 6-9%，on_line 0.26-0.56 — 修的更准
   - v3 review 担心"灰带"，所以本轮主推 `rt=1.0`。

4. **core_residual_v4 整体没怎么下降**：v3 baseline 是 +24，最好的 #1 是 +18.7。
   原因是 cleanup mask 主要修的是 halo 附近残留，而 v3 core 已经被强替换过，
   再被 cleanup 替换一次也是同一个 `bg_low_after` 附近，提升有限。
   **真正的视觉收益在 halo 区域和原 line 断点处**，看 zoom 即可。

5. **`v4_rt1.0_cl31_cw1`（#1）相对 v3 的核心收益**：
   - on_line=0.56 说明 56% 的清理在已知线上（精准）
   - core_v4=+18.7 最低（残留最少）
   - cleanup_cov=9.4%（适中，不会形成大片灰带）

## 怎么挑

- **默认主推 #1**（`v4_rt1.0_cl31_cw1`）—— 三个指标都最好，且 closing 帮助
  延续线方向。
- 重点看 `zoom_compare_sheet.jpg` 的 Z2 / Z3 / Z4——v3 review 说"贯穿长线、
  星芒交叉、右下黑线"还看得见，对照 v3 baseline 行与 v4 #1 行的 crop。
- 看 `baseline_v3_vs_v4_compare.jpg` 直接对比 v3 baseline ↔ v4 主推。
- 如果担心 #1 漏修：用 #4（`rt=0.8_cl31`，扩大覆盖）。
- 如果担心 #1 留下断续黑点：用 #2（`rt=1.0_cl15`，更小 cleanup 但全图 p99 最低）。
- **不建议** #6 / #7 作为主方案 —— `rt=0.5` 是 v3 review 警告的"灰刷痕"区。
