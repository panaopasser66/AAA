# Visual Line Hide v4 — Review Run

## 本轮目标

v3 review 结论：
- v3 是目前最好的一版，**方向性背景采样有效**；
- 但 suppressed 图里仍有**断续黑点 / 黑段**，集中在线交叉点和原始长线位置；
- 激进参数 (`C90b_bt0.5_wc7_h8_kh0.3`) 去线强但出现**灰色刷痕**，**不可作为
  主方案**。

v4 的目标：**在 v3 baseline (`C90b_bt1.0_wc7_h6_kh0.3`) 之上加二次清理**，
把那些断续黑点/黑段进一步擦掉，同时避免大面积灰带。

### v4 新增的 5 个动作

1. **固定 v3 baseline**（不再扫主参数）：mask=C90b, bt=1.0, wc=7, hw=6,
   kc=0.0, kh=0.3, side_sigma=15, tensor_sigma=4。
2. **二次 residual dark score** 在 `suppressed_v3` 上重新计算：
   - `bg_low_after = gaussian(suppressed_v3, σ=20)`
   - `dark_residual_after = bg_low_after - suppressed_v3`
   - `residual_score = (dark_residual_after)+ / max(local_noise, MAD)`
3. **Protect mask 分割**：原 442 个 protect 组件中
   - **48 个 `protect_true_blob`**：远离 line，**继续保护**；
   - **394 个 `protect_on_line`**：与 `dilate(broad_line_mask, 3)` 重叠 > 30%，
     **不再保护**，允许被清理。这就是为什么 v3 在某些"黑点"上下不去刀的原因。
4. **方向性 mask continuity**：对 `v3_core` 做 6 角度（0/30/60/90/120/150°）
   line-SE closing，把沿线方向的断裂补上。新增的像素**只接受落在
   `broad_line_mask` 里的**，避免乱搭桥。
5. **Cleanup 用 v3 同款 orientation-aware bg**（距离改小 {4,7,10,14}），
   两侧采样无效时回退 side-band。Cleanup 区域**硬替换**，无 soft 保留。

### Cleanup 候选的线性约束

- 必须在 `dilate(line_region, 4)` 之内（`line_region = v3_core | v3_halo | broad_line_mask`）
- Connected component 必须满足：
  - 与 `broad_line_mask` 直接重叠（in_line）且 area ≥ 4，**或**
  - 偏心率 > 0.85 且 area ≥ 12（细长结构）
- 小 compact blob 不在线上的直接丢弃

本轮仍然**不做** defect detection / 模型训练 / self-supervised inpainting。

## 运行命令

```bash
python scripts/route_c_bg_sampling_band_suppress/line_band_contrast_suppress_v4.py \
    --input data/data.tif \
    --prepared outputs/prepare \
    --out outputs/visual_line_hide_v4
```

### Sweep 配置（精简到 15）

| 维度 | 取值 |
|---|---|
| residual_threshold | 0.5 / 0.8 / 1.0 |
| closing_len | 0 / 15 / 31 |
| closing_width | 1 / 2（当 closing_len=0 时退化为 1） |
| keep_halo | 0.3（固定，仅控制 v3 baseline 的 halo） |

实际去重后 = 3 × 5 = **15 个变体**。

## 输入路径

- `data/data.tif`（不提交）
- `outputs/prepare/`（不提交）
  - `masks/mask_C_sato90.png`、`mask_F_sato75.png`、`protect_mask.png`
  - `responses/response_sato.tif`

## 输出路径

### 完整产物（不入 Git）

- `outputs/visual_line_hide_v4/v3_baseline_preview.png`
- `outputs/visual_line_hide_v4/residual_dark_score_preview.png`
- `outputs/visual_line_hide_v4/protect_true_blob.png` /
  `outputs/visual_line_hide_v4/protect_on_line.png`
- `outputs/visual_line_hide_v4/reports/`
  - `summary.csv`、`contact_sheet.png`、`compare_sheet.png`、
    `zoom_compare_sheet.png`、`baseline_v3_vs_v4_compare.png`、
    `zoom_crops.json`
- `outputs/visual_line_hide_v4/variants/<name>/`
  - `summary_panel.png`（含 4 个 zoom crop）
  - `suppressed_preview.png`、`removed.png`、`cleanup_overlay.png`
  - `cleanup_mask.png`、`residual_dark_mask.png`、`added_from_closing.png`
  - `suppressed.tif`、`metrics.json`

### 审核可提交产物（本目录，已入 Git）

- **`baseline_v3_vs_v4_compare.jpg`** — 主推 v4 对照 v3 baseline 的全图 + 4 zoom
  三行对照
- **`zoom_compare_sheet.jpg`** — raw / v3 baseline / Top 5 v4 variants × 4 zoom
- `compare_sheet.jpg` — 全部 15 个变体的 cleanup_overlay / suppressed / removed
- `contact_sheet.jpg` — 全部 15 个 suppressed 缩略图
- `v3_baseline_preview.jpg` —— 共享的 v3 baseline 全图
- `residual_dark_score_preview.jpg` —— 在 v3 baseline 上重算的 residual 热图
- `protect_true_blob.jpg` —— 远离 line 的 protect blob（继续保护）
- `protect_on_line.jpg` —— 落在 line 上的 protect blob（允许清理，**大头**）
- `summary.csv`、`shortlist.md`、`README.md`
- `variants/<top>/{summary_panel,suppressed_preview,removed,cleanup_mask}.jpg`
  × 7

所有 JPG 最长边 ≤ 2800 px，quality=85。

## 重点查看文件

1. **`baseline_v3_vs_v4_compare.jpg`** —— 本轮最关键的一张。三行：
   raw / v3 baseline / v4 主推；每行 5 张（全图 + 4 zoom）。直接看 v4 主推
   是否真的把 v3 残留的黑点/黑段擦掉了。
2. **`zoom_compare_sheet.jpg`** —— Top 5 variants × 4 zoom，附 raw + v3 baseline
   作前两行。重点看 Z2/Z3/Z4 的线交叉点。
3. **`shortlist.md`** + `variants/<top>/summary_panel.jpg` —— 每个 panel 上半行
   raw / v3 / v4 / cleanup_mask overlay，下半行 4 个 v4 zoom。
4. **`variants/<top>/cleanup_mask.jpg`** —— cleanup 究竟改了哪些像素。
   稀疏的细长形 = 好；大面积块状 = 警惕灰带。
5. **`protect_on_line.jpg`** —— 检查 v4 把多少原来 "protect" 的小 blob 放给
   cleanup 处理。如果上面有真正的小缺陷被释放了，需要回去调
   `overlap_threshold` (默认 0.3)。
6. **`summary.csv` 重点列**：
   - `cleanup_coverage_percent`（理想 5-10%）
   - `cleanup_on_line_fraction`（越高越精准）
   - `residual_p99_v4` 对比 `residual_p99_v3=2.65`

## Top variants

| # | variant_name | cleanup_cov% | on_line | p99_v4 | core_v4 | 备注 |
|---|---|---:|---:|---:|---:|---|
| 1 | `v4_rt1.0_cl31_cw1` |  9.39 | **0.56** | 2.58 | **+18.7** | **主推** |
| 2 | `v4_rt1.0_cl15_cw1` |  6.87 | 0.40 | **2.56** | +21.3 | 全图 p99 最低 |
| 3 | `v4_rt1.0_cl0`      |  5.57 | 0.26 | 2.56 | +19.7 | 最保守 |
| 4 | `v4_rt0.8_cl31_cw1` | 11.18 | 0.51 | 2.61 | +19.6 | 扩大覆盖 |
| 5 | `v4_rt0.8_cl15_cw1` |  8.74 | 0.38 | 2.60 | +21.4 | 中庸 |
| 6 | `v4_rt0.5_cl31_cw1` | 14.91 | 0.46 | 2.67 | +20.9 | **激进**（注意灰带） |
| 7 | `v4_rt0.5_cl0`      | 11.45 | 0.29 | 2.66 | +19.7 | **最激进**（最易乱修） |

详见 `shortlist.md`。

## 本轮明确不做

- ❌ 不做 defect detection
- ❌ 不训练任何模型（Route B U-Net / Route D inpainting U-Net 都不跑）
- ❌ 不回到 self-supervised inpainting
- ❌ 不修改 `main` 分支；所有改动只在 worktree
  `experiment/visual-line-hide-v1`
- ❌ 不提交 `data/data.tif`、`outputs/`、任何 `.tif/.npz/.npy/.pt` 等大文件

v4 全部任务是 "在 v3 baseline 上做一次精准的二次清理，把残留的断续黑点压平"。
