# Visual Line Hide v5 UNet Refine — Top Variant Shortlist

来自 18 个 refine 变体（基线固定为 `v5_ct0.50_ht0.45_cl0` mask）。

## 关键观察

混合公式：`out = raw * (1 - gamma) + bg * gamma`

- **hard_core mode**: gamma 在 core 整体均匀（= `core_mix`），halo 均匀（= `1 - keep_halo`）
- **distance_core mode**: gamma 在 core 内从边界的 `1 - keep_halo` 平滑过渡到中心的 `core_mix`
- 当 `core_mix == 1 - keep_halo` 时 (`cm=0.7,kh=0.3`)，hc 和 dc 退化为同一结果

`rm_core` 大小（线被压走的强度）正比于 `gamma_mean_core`：
- gamma_core ≈ 0.7 → rm_core ≈ 215（弱）
- gamma_core ≈ 0.85 → rm_core ≈ 261（中）
- gamma_core ≈ 1.00 → rm_core ≈ 308（强，等于 v5_unet baseline）

## 18 个变体全表

| variant | mode | cm | kh | rm_core | rm_halo | gm_core | gm_halo |
|---|---|---:|---:|---:|---:|---:|---:|
| v5r_cm0.70_kh0.3_hc | hc | 0.70 | 0.3 | 215.3 | 97.1 | 0.70 | 0.70 |
| v5r_cm0.70_kh0.3_dc | dc | 0.70 | 0.3 | 215.3 | 97.1 | 0.70 | 0.70 |
| v5r_cm0.70_kh0.5_hc | hc | 0.70 | 0.5 | 215.3 | 69.4 | 0.70 | 0.50 |
| v5r_cm0.70_kh0.5_dc | dc | 0.70 | 0.5 | 195.8 | 69.4 | 0.61 | 0.50 |
| v5r_cm0.70_kh0.7_hc | hc | 0.70 | 0.7 | 215.3 | 41.6 | 0.70 | 0.30 |
| v5r_cm0.70_kh0.7_dc | dc | 0.70 | 0.7 | 176.4 | 41.6 | 0.52 | 0.30 |
| v5r_cm0.85_kh0.3_hc | hc | 0.85 | 0.3 | 261.4 | 97.1 | 0.85 | 0.70 |
| **v5r_cm0.85_kh0.3_dc** | dc | 0.85 | 0.3 | 246.8 | 97.1 | 0.78 | 0.70 |
| v5r_cm0.85_kh0.5_hc | hc | 0.85 | 0.5 | 261.4 | 69.4 | 0.85 | 0.50 |
| **v5r_cm0.85_kh0.5_dc** | dc | 0.85 | 0.5 | 227.4 | 69.4 | 0.70 | 0.50 |
| v5r_cm0.85_kh0.7_hc | hc | 0.85 | 0.7 | 261.4 | 41.6 | 0.85 | 0.30 |
| v5r_cm0.85_kh0.7_dc | dc | 0.85 | 0.7 | 207.9 | 41.6 | 0.61 | 0.30 |
| v5r_cm1.00_kh0.3_hc | hc | 1.00 | 0.3 | **307.6** | 97.1 | 1.00 | 0.70 | ← v5_unet 等价 |
| **v5r_cm1.00_kh0.3_dc** | dc | 1.00 | 0.3 | 278.4 | 97.1 | 0.87 | 0.70 | ← **主推** |
| v5r_cm1.00_kh0.5_hc | hc | 1.00 | 0.5 | 307.6 | 69.4 | 1.00 | 0.50 |
| **v5r_cm1.00_kh0.5_dc** | dc | 1.00 | 0.5 | 258.9 | 69.4 | 0.78 | 0.50 |
| v5r_cm1.00_kh0.7_hc | hc | 1.00 | 0.7 | 307.6 | 41.6 | 1.00 | 0.30 |
| v5r_cm1.00_kh0.7_dc | dc | 1.00 | 0.7 | 239.5 | 41.6 | 0.69 | 0.30 |

> 所有 `rm_outside = 0.00`（编辑严格限定在 core+halo 内）。
> texture_scale = 0.0 for all 18 variants (本轮固定，未启用纹理保留)。

## 7 个候选

| # | variant_name | rm_core | gm_core | gm_halo | 设计意图 |
|---|---|---:|---:|---:|---|
| 1 | **`v5r_cm1.00_kh0.3_dc_ts0.00`** | 278 | 0.87 | 0.70 | **主推**：保留 v5_unet 强度，core 中心→edge 平滑过渡 |
| 2 | `v5r_cm0.85_kh0.3_dc_ts0.00` | 247 | 0.78 | 0.70 | 比 #1 软 10%（cm=0.85） |
| 3 | `v5r_cm0.85_kh0.5_dc_ts0.00` | 227 | 0.70 | 0.50 | halo 更软（kh=0.5），减轻 halo 刷痕 |
| 4 | `v5r_cm1.00_kh0.5_dc_ts0.00` | 259 | 0.78 | 0.50 | 全力 core + 软 halo（推荐用于刷痕严重区） |
| 5 | `v5r_cm0.85_kh0.5_hc_ts0.00` | 261 | 0.85 | 0.50 | hard_core 对照（无 distance 渐变） |
| 6 | `v5r_cm0.70_kh0.5_dc_ts0.00` | 196 | 0.61 | 0.50 | 最软 core；看是否保留过多 raw |
| 7 | `v5r_cm1.00_kh0.7_dc_ts0.00` | 240 | 0.69 | 0.30 | halo 最软（kh=0.7 保留 70% raw），可能最自然 |

## 排序原理

本轮 review 重点不是"线压多狠"，而是"背景是否自然"。所以：

- **`core_mix` 越低**，core 保留越多 raw → 视觉上线被压弱但更自然
- **`keep_halo` 越高**，halo 保留越多 raw → 刷痕越少
- **`distance_core` 让 core 边缘和 halo 平滑相接**，避免硬边
- 主推 #1 (`cm=1.0, kh=0.3, dc`)：保留全 core 强度但给 core 一个内部 alpha
  渐变，让边界不会突变（v5_unet baseline 等价是 hc 同参数 → 硬边）

## v5_unet baseline vs refine 主推

| | v5_unet baseline | **refine #1** |
|---|---:|---:|
| 变体 | `v5_ct0.50_ht0.45_cl0` | `v5r_cm1.00_kh0.3_dc_ts0.00` |
| Mask | line_prob > 0.5 (core) + >0.45 dilated (halo) | **完全相同** |
| core gamma 形态 | 整体 1.0 (硬边) | edge 0.70 → center 1.0 (距离渐变) |
| 平均 gamma_core | 1.00 | 0.87 |
| rm_core | 307.6 | 278.4 |
| 设计目的 | 把线压到底 | 把线压到底但 core 边缘和 halo 平滑相接 |

主要预期改进：v5_unet 在 core/halo 交界处的硬边消失，过渡变得自然，应当
减轻肉眼可见的刷痕。

## 怎么挑

- **默认 #1** —— 最接近 v5_unet 主推但软化了 core 边缘。
- 看 `baseline_compare.jpg` 直接对比 raw / v5_unet / refine #1。
- 看 `zoom_compare_sheet.jpg` 比较 Top 5。
- 如果 #1 仍有刷痕，下一档试 #3 或 #4（kh=0.5 软 halo）。
- 如果担心线被压不彻底，验证 #1 的 zoom 是否仍能看到长线残留。
- **不建议** #5（hc 硬边，等价于另一种 v5_unet 副本）— 没有 refine 价值。
- **不建议** #6（cm=0.7 过软）— rm_core 跌到 196，预期线又看得见了。

## 下一步可选项（如果 #1 还不够好）

1. 加入 `texture_scale = 0.15` 重跑 Top 3 — 给 bg 加点邻域纹理变化，去掉
   "完全平整的修补感"。
2. 进一步细化 `distance_max`（当前 5），更小 distance_max → 渐变区更窄、
   只在 core 边 2-3 像素软化；更大 → 整个 core 都有过渡。
