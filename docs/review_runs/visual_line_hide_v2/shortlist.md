# Visual Line Hide v2 — Top Variant Shortlist

来自 144 个 v2 变体的 `summary.csv` 初筛。

## 关于指标

- `core_cov%`：核心 mask（强替换区）占比
- `edit_cov%`：核心 + halo 总编辑占比
- `rm_core`：核心区平均 `|raw - suppressed|`（值越大代表线被压得越彻底）
- `rm_halo`：halo 区平均 `|raw - suppressed|`（halo 是过渡带）
- `rm_out`：编辑区外平均 `|raw - suppressed|`（几乎为零，因为 v2 的 feather +
  side-band bg 把扰动严格限制在编辑区）

> ⚠️ v2 的 `core_outside_ratio` 数值上是 `~1e8` 量级，因为 `rm_out` 接近 0；
> 这是“几乎不漏到外面”的反映，不是真有意义的相对量，**别把它当排序键**。
> 选型主要看 `rm_core` + `core_cov%` + 视觉效果。

## 挑选原则

v1 的问题是“主黑线在 suppressed 图里仍然明显”，所以 v2 主要看：

1. `keep_core = 0.00` 的最激进核心替换（线核心几乎完全被 `bg` 覆盖）
2. core 不要太窄（v1 的 width=2 留下了 residual），优先 `wc=5` 或 `wc=7`
3. halo 给一个 `wc + halo` 的明显宽度过渡
4. C_sato90 的 mask 较窄但精确；D_sato85 的 mask 更广（可覆盖较粗的线）

## 7 个候选

| # | variant_name | core_cov% | edit_cov% | rm_core | rm_halo | 说明 |
|---|---|---:|---:|---:|---:|---|
| 1 | `C90b_wc5_h8_kc0.00_kh0.3_s15` | 5.14 | 18.85 | 233.7 | 145.7 | **主推**：窄核心 + 宽 halo + 强核心替换 |
| 2 | `C90b_wc7_h8_kc0.00_kh0.3_s15` | 8.02 | 22.34 | 236.0 | 139.8 | wc=7 处理粗线；rm_core 最大 |
| 3 | `C90b_wc5_h4_kc0.00_kh0.3_s15` | 5.14 | 11.35 | 223.6 | 150.0 | 更紧的 halo，编辑区小 |
| 4 | `C90b_wc5_h8_kc0.05_kh0.3_s15` | 5.14 | 18.85 | 222.0 | 145.7 | kc=0.05 留 5% 核心残差作对照 |
| 5 | `C90b_wc5_h8_kc0.00_kh0.2_s15` | 5.14 | 18.85 | 233.7 | 166.5 | kh=0.2 让 halo 也更强压制 |
| 6 | `D85b_wc5_h8_kc0.00_kh0.3_s15` | 9.92 | 31.86 | 226.0 | 133.0 | D 系 mask，捕捉更广的线 |
| 7 | `D85b_wc7_h8_kc0.00_kh0.3_s15` | 14.98 | 36.80 | 224.7 | 127.6 | D + wc=7：最激进覆盖 |

## v1 → v2 主要差异

| 维度 | v1 | v2 |
|---|---|---|
| Mask 来源 | 仅 sato 百分位 | sato ∩ `black_line_score > 1.0`（剔除亮脊） |
| Mask 结构 | 单 line_band（一档 keep） | core + halo 双层，独立 keep |
| Core keep | 0.0 / 0.1 / 0.2，但 feather 会把 raw 重新拼回核心 | core 在 feather 后**再次硬覆盖**为 `bg + kc·residual`，保证“核心真正被替换” |
| 背景估计 | `masked_gaussian` 或 `side_band`，自动 fallback 大 sigma | 仅 `side_band`，禁用全图回退 |
| black-line score | ❌ | ✅ `(bg_low - raw)+ / local_noise`，先存一张 `black_line_score_preview.jpg` 供检查 |

## 下一步建议

- 主看 #1 (`C90b_wc5_h8_kc0.00_kh0.3_s15`) 的 `suppressed_preview.jpg`：
  主黑线在 v2 下是否已经基本融入背景？
- 如果细线段仍残留，去看 #2 (wc=7) 或 #6/#7 (D85b)。
- 如果 halo 看起来有“晕开”，对比 #1 vs #5 (kh=0.2 vs 0.3) 或 #3 (h=4)。
- `removed.jpg` 应该几乎只在核心 + halo 内显色；外部应该是中性灰。
