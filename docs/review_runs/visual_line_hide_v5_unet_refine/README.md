# Visual Line Hide v5 UNet Refine — Review Run

## 本轮目标

v5_unet review 结论：**当前最佳 baseline = `v5_ct0.50_ht0.45_cl0`**（去黑线
最强的一版），但**局部出现明显刷痕 / 亮带 / 背景过度修改**。

> 下一步不再扩 mask，也不做更强 cleanup。

本轮目标：**在保持去线效果的同时，让背景更自然**。固定 v5_unet baseline 的
mask（line_prob > 0.5 / > 0.45 dilated, no closing），**只在 suppression 的
应用方式上小范围 refine**。

本轮**不做** defect detection，**不训练**，**不改 mask**。

## 4 个新动作

1. **Soft core replacement** —— 把硬替换
   `out_core = bg_dir`
   换成
   `out_core = raw * (1 - core_mix) + bg_dir * core_mix`，
   `core_mix ∈ {0.70, 0.85, 1.00}`。`cm=1.0` 等于原 v5_unet 硬替换，
   更低值保留部分 raw。

2. **Halo attenuation 调弱** —— `keep_halo ∈ {0.3, 0.5, 0.7}`，让 halo
   保留更多原始纹理，减轻"刷痕"感。

3. **Distance-based alpha (`mode=distance_core`)** ——
   `alpha = clip(distance_inside_core / max_dist, 0, 1)`，core 中心
   `gamma=core_mix`，core 边缘 `gamma=1 - keep_halo`（与 halo 平滑相接）。
   `mode=hard_core` 则是 core 内整体均匀的 gamma，作对照。

4. **Background texture preservation (optional)** ——
   `bg = bg_dir + texture_scale * local_texture`，其中
   `local_texture = gaussian(raw - gaussian(raw, σ=20), σ=8)` 在线外区域估计
   后扩散。本轮 `texture_scale = 0`（推迟到下一轮）。

## 关键公式

```
out = raw * (1 - gamma) + bg * gamma

outside edit:     gamma = 0
halo:             gamma = 1 - keep_halo
core (hard_core): gamma = core_mix
core (distance_core):
    edge_gamma  = 1 - keep_halo
    alpha       = clip(distance_inside_core / 5.0, 0, 1)
    gamma       = lerp(edge_gamma, core_mix, alpha)
```

distance_core 保证 core 边缘 (`alpha→0`) 的 gamma 等于 halo 的 gamma，
所以两者在边界**连续**，没有硬跳。

## 运行命令

```bash
python scripts/route_c_bg_sampling_band_suppress/line_band_contrast_suppress_v5_unet_refine.py \
    --input data/data.tif \
    --prepared outputs/prepare \
    --line-prob /e/dapro/dapro-deep-mask-v1/outputs/route_b_line_mask_v1/line_prob.tif \
    --out outputs/visual_line_hide_v5_unet_refine \
    --v5unet-baseline-preview \
        outputs/visual_line_hide_v5_unet/variants/v5_ct0.50_ht0.45_cl0/suppressed_preview.png
```

### 固定的 baseline mask（与 v5_unet 主推完全相同）

| | |
|---|---|
| line_prob | Route B v1 UNet 输出 |
| core_th | 0.50 |
| halo_th | 0.45 |
| halo_dilate | 2 |
| closing_len | 0 |
| core_cov | 22.59% |
| edit_cov | 33.15% |

### Refine sweep

| 维度 | 取值 |
|---|---|
| core_mix | 0.70 / 0.85 / 1.00 |
| keep_halo | 0.3 / 0.5 / 0.7 |
| mode | hard_core / distance_core |
| texture_scale | 0.0（固定） |

3 × 3 × 2 × 1 = **18 个变体**，~80 秒跑完。

## 输入路径

- `data/data.tif` （不进 Git）
- `outputs/prepare/`（不进 Git）
- `/e/dapro/dapro-deep-mask-v1/outputs/route_b_line_mask_v1/line_prob.tif`
- `outputs/visual_line_hide_v5_unet/variants/v5_ct0.50_ht0.45_cl0/suppressed_preview.png`
  （v5_unet baseline 参考；refine 脚本会用作 baseline_compare 的中间行）

## 输出路径

### 完整产物（不入 Git）

- `outputs/visual_line_hide_v5_unet_refine/`
  - `v5_unet_baseline_preview.png`、`core_mask.png`、`halo_mask.png`、
    `protect_true_blob.png`、`protect_on_line.png`
  - `reports/{summary.csv, contact_sheet.png, compare_sheet.png,
    zoom_compare_sheet.png, baseline_compare.png, zoom_crops.json}`
  - `variants/<name>/`
    - `summary_panel.png`（含 4 个 zoom + gamma 可视化）
    - `suppressed_preview.png`、`removed.png`、`gamma_map.png`、
      `suppressed.tif`、`metrics.json`

### 审核可提交产物（本目录，已入 Git）

- `README.md`、`shortlist.md`、`summary.csv`
- **`baseline_compare.jpg`** —— 3 行：raw / v5_unet baseline / refine 主推；
  每行 5 张（full + 4 zoom）
- **`zoom_compare_sheet.jpg`** —— raw / v5_unet baseline / refine Top 5
  × 4 zoom 区域
- `compare_sheet.jpg` —— 18 个变体的 gamma / suppressed / removed 三列
- `contact_sheet.jpg` —— 18 个 suppressed 缩略
- `v5_unet_baseline_preview.jpg` —— 中间参考
- `variants/<top>/{summary_panel, suppressed_preview, removed}.jpg` × 7

所有 JPG 最长边 ≤ 2800 px，quality=85。

## 重点查看文件

1. **`baseline_compare.jpg`** —— 本轮**最关键的一张**。直接对比 raw 与
   v5_unet baseline 与 refine 主推在 4 个 zoom 上的视觉效果。
   重点判断：**refine 主推是否在保持去线的同时让背景更自然 / 刷痕更少**。
2. `zoom_compare_sheet.jpg` —— Top 5 refine 变体 vs raw / v5_unet baseline。
3. `shortlist.md` —— 18 行表格 + 7 个候选 + 设计意图。
4. `compare_sheet.jpg` —— 18 行 (gamma 图 / suppressed / diff)。
5. `variants/v5r_cm1.00_kh0.3_dc_ts0.00/summary_panel.jpg` —— 主推变体细节。
6. `summary.csv` 重点列：
   - `gamma_mean_core`、`gamma_mean_halo`：实际平均 gamma（distance_core 下
     core 平均小于 core_mix）
   - `mean_abs_removed_core`：实际线压走强度（vs v5_unet baseline 307.6）

## Top 7 候选（手挑，按对刷痕的预期减少程度）

| # | variant_name | rm_core | gm_core | gm_halo | 备注 |
|---|---|---:|---:|---:|---|
| 1 | **`v5r_cm1.00_kh0.3_dc_ts0.00`** | 278 | 0.87 | 0.70 | **主推**：全 core 强度 + edge 平滑 |
| 2 | `v5r_cm0.85_kh0.3_dc_ts0.00` | 247 | 0.78 | 0.70 | core 软 10% |
| 3 | `v5r_cm0.85_kh0.5_dc_ts0.00` | 227 | 0.70 | 0.50 | core 软 + halo 软 |
| 4 | `v5r_cm1.00_kh0.5_dc_ts0.00` | 259 | 0.78 | 0.50 | 全 core + 软 halo |
| 5 | `v5r_cm0.85_kh0.5_hc_ts0.00` | 261 | 0.85 | 0.50 | hc 对照（无 distance 渐变） |
| 6 | `v5r_cm0.70_kh0.5_dc_ts0.00` | 196 | 0.61 | 0.50 | 最软 core 对照（可能线又看见） |
| 7 | `v5r_cm1.00_kh0.7_dc_ts0.00` | 240 | 0.69 | 0.30 | halo 最软（kh=0.7） |

详见 `shortlist.md`。

## 本轮明确不做

- ❌ 不做 defect detection
- ❌ 不改 v5_unet mask（core_th / halo_th / closing_len / halo_dilate
  全部固定）
- ❌ 不训练模型；不重跑 Route B
- ❌ 不修改 main 分支；所有改动只在 worktree `experiment/visual-line-hide-v1`
- ❌ 不提交 `data/`、`outputs/`、`.tif/.npz/.npy/.pt` 等大文件
- ⏸ `texture_scale` 沿用 0.0（不启用纹理保留）；待 review 后决定是否在 Top
  上跑 `texture_scale=0.15` 二阶段。

refine 的全部任务是 **在 v5_unet 已选定的 mask 上，让 suppression 应用方式
更柔和、过渡更自然**。
