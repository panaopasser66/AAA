# Visual Line Hide v1 — Review Run

## 本轮目标

对 X-ray 图像中已经 mask 出来的 Kossel 黑线区域**重新赋值**，降低这些线相对周围
背景的 contrast，让黑线**视觉上隐藏进背景**。

本轮**不做** defect detection，**不训练模型**，**不做复杂 inpainting**。仅使用
Route C（局部背景估计 + 线带 contrast 抑制）的经典管线，输出可手工挑选的多组变体。

## 运行命令

在 worktree `e:/dapro/dapro-visual-line-hide-v1`、分支
`experiment/visual-line-hide-v1` 下：

```bash
# 1) 准备 mask、response、protect_mask、global_arrays 等
python scripts/route_d_selfsupervised_inpainting/prepare_project2_data.py \
    --input data/data.tif \
    --out outputs/prepare

# 2) 线带 contrast 抑制（108 个变体扫描）
python scripts/route_c_bg_sampling_band_suppress/line_band_contrast_suppress.py \
    --input data/data.tif \
    --prepared outputs/prepare \
    --out outputs/visual_line_hide_v1
```

扫描维度（默认参数）：
- mask：`B_sato95` / `C_sato90` / `D_sato85`
- 线带宽度 width：2 / 4 / 6
- 背景估计方法：`masked_gaussian` / `side_band`
- sigma：15 / 25
- keep_ratio：0.0 / 0.1 / 0.2
- feather_radius：2

共 3 × 3 × 2 × 2 × 3 = **108** 个变体。

## 输入路径

- 原始图：`data/data.tif`（被 `.gitignore` 忽略，不进 Git）
- prepare 输出：`outputs/prepare/`（被 `.gitignore` 忽略，不进 Git）
  - `masks/mask_*.png`：Kossel 线 mask（按 sato 百分位阈值）
  - `masks/protect_mask.png`：保护小 blob 不被涂掉
  - `responses/response_sato.tif`：sato 多尺度响应

## 输出路径

### 完整产物（不入 Git）

- `outputs/visual_line_hide_v1/variants/<variant_name>/`
  - `suppressed.tif`、`background_estimate.tif`、`metrics.json`
  - `summary_panel.png` / `suppressed_preview.png` / `overlay.png` /
    `removed.png` / `edit_mask.png` / `line_band.png`
- `outputs/visual_line_hide_v1/reports/`
  - `contact_sheet.png`、`compare_sheet.png`、`compare_sheet_all.png`、
    `summary.csv`

### 审核可提交产物（本目录，已入 Git）

- `compare_sheet.jpg` — `side_band, sigma=15` 子集的三列对比（27 行）
- `contact_sheet.jpg` — 全部 108 个变体的 suppressed 缩略图
- `summary.csv` — 每个变体的 coverage / in-out removed-ratio 等指标
- `shortlist.md` — 初筛 Top 7 variants（指标 + 选型建议）
- `variants/<name>/summary_panel.jpg|suppressed_preview.jpg|removed.jpg`
  — Top variant 的三类核心图

所有 JPG 最长边 ≤ 2800px，quality=85。

## 重点查看文件

1. **`contact_sheet.jpg`** — 全局快速扫一遍 108 个变体的 suppressed 缩略图。
2. **`compare_sheet.jpg`** — 在 `side_band, sigma=15` 子集（27 行）里逐行看
   `overlay` → `suppressed` → `removed`：
   - 黑线是否消失到看不出？
   - `removed` 热度图是否只在线带内集中？（外部不应有显著 diff）
3. **`shortlist.md`** — 7 个候选变体的指标排序与选型建议。
4. **`variants/<top_name>/summary_panel.jpg`** — 候选变体的四联对照
   （raw / edit_mask / suppressed / diff）。
5. **`variants/<top_name>/suppressed_preview.jpg`** — 候选变体抑制后的全分辨率图。
6. **`summary.csv`** 中重点列：
   - `edit_coverage_percent`（编辑面积）
   - `mean_abs_removed_inside` / `mean_abs_removed_outside`
   - `removed_inside_outside_ratio`（越大代表越「就线论线」）

## Top variants

初筛后保留 7 个变体（见 `shortlist.md`）：

| # | variant_name | cov% | ratio | 选型场景 |
|---|---|---:|---:|---|
| 1 | `B_sato95_width6_mg_sigma15_keep0.1`  | 18.83 | **100.12** | 最干净，外部扰动近零 |
| 2 | `B_sato95_width4_mg_sigma15_keep0.1`  | 14.91 | 90.47  | cov 最小、ratio 高 |
| 3 | `B_sato95_width4_side_sigma15_keep0.1` | 14.91 | 75.76 | side_band 在 B mask 下的对照 |
| 4 | `C_sato90_width6_side_sigma15_keep0.1` | 32.88 | 50.93 | 中等覆盖 + side_band |
| 5 | `C_sato90_width4_side_sigma15_keep0.1` | 26.62 | 49.40 | 中等覆盖 + side_band |
| 6 | `D_sato85_width6_side_sigma15_keep0.1` | 45.55 | 34.50 | 激进覆盖，最广义隐线 |
| 7 | `D_sato85_width4_side_sigma15_keep0.1` | 37.39 | 34.35 | 激进覆盖，最广义隐线 |

(`cov%` = `edit_coverage_percent`；`ratio` = inside/outside mean-abs-removed ratio。)

下一步：人工从这 7 个里挑 1–2 个作为 hand-picked 输出。

## 本轮明确不做

- ❌ 不做 defect detection
- ❌ 不训练任何模型（Route B U-Net / Route D inpainting U-Net 都不跑）
- ❌ 不做复杂 inpainting（PatchMatch、扩散模型、深度 inpainting 等）
- ❌ 不修改 `main` 分支；所有改动只在本 worktree
  `experiment/visual-line-hide-v1`
- ❌ 不提交 `data/data.tif`、`outputs/`、任何 `.tif/.npz/.npy/.pt` 等大文件

仅评估「contrast suppression 一招能把黑线隐去多远」。
