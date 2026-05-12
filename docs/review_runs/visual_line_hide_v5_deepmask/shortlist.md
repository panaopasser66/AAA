# Visual Line Hide v5 (deepmask) — Top Variant Shortlist

来自 10 个 v5 变体的初筛。v5 用 **line_prob** 作为 mask 来源，配 v4 的
orientation-aware bg。本轮的 line_prob 来自**合成 fallback**：

```
raw = sato_norm * black_line_score
prob = sigmoid(6.0 * raw - 2.0)     # gain=6, bias=2
prob = gaussian(prob, σ=0.8)
```

合成 line_prob 分布：`p50=0.14, p90=0.26, p99=0.57`。
> ⚠️ 这是 placeholder。真正的 Route B U-Net 给出的 prob 应在线带内饱和 ~1.0，
> 这样 `core_th=0.5` 可以覆盖更宽的核心区。

## 关键指标

- `core%`：`line_prob > core_threshold` 的覆盖率
- `edit%`：`(line_prob > halo_threshold)` 经 dilation 后的总编辑覆盖率
- `dir%`：核心区中通过方向性 bg 成功取到样本的比例
- `core_resid`：核心区抑制后相对 `bg_low(raw)` 的残留 darkness
  （**v3/v4 的同位指标在 +18~+25 量级；v5 主推压到 -0.60**）

## 7 个候选

| # | variant_name | core% | edit% | dir% | core_resid | 备注 |
|---|---|---:|---:|---:|---:|---|
| 1 | `v5_ct0.50_ht0.35_cl0`  | 1.58 | 14.07 | 69.7 | **-0.60** | **主推**：core_resid 最接近 0 |
| 2 | `v5_ct0.50_ht0.35_cl15` | 3.95 | 14.77 | 66.5 | -5.02 | +方向性 closing；core 翻倍 |
| 3 | `v5_ct0.60_ht0.35_cl0`  | 0.79 | 14.07 | 64.0 | -7.32 | 更紧 core，最稀疏 cleanup |
| 4 | `v5_ct0.60_ht0.35_cl15` | 1.76 | 14.29 | 59.8 | -8.76 | 上面 +closing |
| 5 | `v5_ct0.50_ht0.25_cl15` | 3.95 | 30.97 | 52.6 | -67.74 | **激进 halo**；edit 占 31%，可能出灰带 |
| 6 | `v5_ct0.60_ht0.45_cl15` | 1.76 |  7.67 | 67.5 | +34.82 | 最紧 halo |
| 7 | `v5_ct0.70_ht0.45_cl15` | 0.70 |  7.43 | 61.2 | +35.64 | 最保守，core 最窄 |

## v3 / v4 / v5 主推对比

| | v3 主推 | v4 主推 | v5 主推 (合成 prob) |
|---|---:|---:|---:|
| 变体 | `C90b_bt1.0_wc7_h6_kh0.3` | `v4_rt1.0_cl31_cw1` | `v5_ct0.50_ht0.35_cl0` |
| Mask 来源 | sato + black_score 规则 | 同 v3 + cleanup pass | **line_prob (合成)** |
| core_cov% | 8.02 | 8.02 | **1.58** |
| edit_cov% | 18.55 | 8.02+halo+cleanup | 14.07 |
| dir bg cov% (core) | 59.3 | 56.7 | **69.7** |
| core_resid | +24 (v4 表里) | +18.7 (v4 表里) | **-0.60** |

## 主要观察

1. **v5 的 core 比 v3/v4 窄得多**（1.58% vs 8.02%）—— 因为合成 line_prob 在
   `> 0.5` 上只覆盖了最强的 ridge 顶部，没有捕捉整条暗线的宽度。
2. **但是在它覆盖到的那部分像素上，方向性 bg 把 core_resid 压到 -0.60** ——
   pipeline 工作得很好。
3. **要让 v5 真正超过 v4，关键是更好的 line_prob**：
   - 用 Route B 的 U-Net 推理产出 prob.tif，直接 `--line-prob path/to/prob.tif`
     注入，**脚本接口已经准备好**。
   - 或调 synthesis（`--synth-logit-gain/bias`，本版暂未暴露 CLI 但函数已分离）
     让 prob 在线带内宽到接近 1.0。
4. **closing 在 v5 的作用不像 v4 那么显著**：增加 core_cov 但不一定改善
   core_resid（因为合成 prob 的位置就不准）。

## 怎么挑

- **如果只用合成 prob，主推 #1** (`v5_ct0.50_ht0.35_cl0`) ——
  core_resid 最接近 0；但 core 窄，整体线消失程度可能不如 v4。
- **不建议把 v5 当成"最终方案"**。v5 是 **架构 demo**：
  1. 跑通 line_prob → core/halo → orientation bg 的全链路；
  2. 验证接 U-Net 的接口可用；
  3. 量化"line_prob 质量"是 v5 vs v4 的瓶颈。
- 下一步：训练 Route B U-Net（或重训）→ 导出 prob.tif → 直接重跑本脚本。
- 看 `zoom_compare_sheet.jpg` 比较 raw 与 5 个 v5 变体的 4 个 zoom region，
  特别关注：合成 prob 在哪里能 catch 线、在哪里 miss（典型 miss：
  星芒交叉点、灰度过渡处）。
