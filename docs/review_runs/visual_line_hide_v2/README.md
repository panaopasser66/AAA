# Visual Line Hide v2 — Review Run

## 本轮目标

v1 的结论：removed 热图集中在线上是对的，但 `suppressed` 里**贯穿主黑线仍然明显**
——核心区在 feather 作用下被 raw 重新覆盖回去了，所以并没有真正消失。

v2 的目标：**把肉眼可见的黑色 Kossel line 进一步隐藏进背景**，做法：

1. **black-line response** —— 用 `(bg_low - raw)/local_noise` 计算
   "this pixel is darker than its surroundings, by how many σ"，
   与 sato mask 求交，剔除亮脊，得到 black-aware 的核心。
2. **core + halo 双层 mask** —— 核心区强替换（`keep_core` 0.0 / 0.05 / 0.1），
   halo 区轻压制（`keep_halo` 0.2 / 0.3），boundary 用 feather 缓接外部。
3. **side-band background** —— 仅用线两侧的环带做背景估计，**不再回退到全图
   Gaussian**。
4. **核心区在 feather 之后再硬覆盖一次**，保证"核心真的被换掉"——这是 v1 的
   主要修复点。

本轮仍然**不做** defect detection，**不训练模型**，**不做复杂 inpainting**。

## 运行命令

```bash
# 1) 准备阶段同 v1（如果 outputs/prepare 还在就跳过）
python scripts/route_d_selfsupervised_inpainting/prepare_project2_data.py \
    --input data/data.tif \
    --out outputs/prepare

# 2) v2 sweep（144 个变体，~19 分钟）
python scripts/route_c_bg_sampling_band_suppress/line_band_contrast_suppress_v2.py \
    --input data/data.tif \
    --prepared outputs/prepare \
    --out outputs/visual_line_hide_v2
```

扫描维度（默认参数）：

| 维度 | 取值 |
|---|---|
| mask | `C_sato90` ∩ black-aware (`C90b`), `D_sato85` ∩ black-aware (`D85b`) |
| width_core | 3 / 5 / 7 |
| halo_width | 4 / 8 |
| keep_core | 0.00 / 0.05 / 0.10 |
| keep_halo | 0.2 / 0.3 |
| side_sigma | 10 / 15 |
| feather_radius | 2（固定） |
| side_radius | 40（固定） |
| black_threshold | 1.0（固定）—— `black_line_score > 1.0` 才进 core |

共 2 × 3 × 2 × 3 × 2 × 2 = **144** 个变体。

## 输入路径

- 原始图：`data/data.tif`（被 `.gitignore` 忽略，不进 Git）
- prepare 输出：`outputs/prepare/`
  - `masks/mask_C_sato90.png`、`masks/mask_D_sato85.png`
  - `masks/protect_mask.png`
  - `responses/response_sato.tif`

## 输出路径

### 完整产物（不入 Git）

- `outputs/visual_line_hide_v2/black_line_score_preview.png`
- `outputs/visual_line_hide_v2/reports/{contact_sheet,compare_sheet,
  compare_sheet_all}.png` + `summary.csv` + `black_line_score_stats.json`
- `outputs/visual_line_hide_v2/variants/<name>/`
  - `summary_panel.png`、`suppressed_preview.png`、`removed.png`
  - `overlay.png`、`core_mask.png`、`halo_mask.png`
  - `suppressed.tif`、`background_estimate.tif`、`metrics.json`

### 审核可提交产物（本目录，已入 Git）

- `compare_sheet.jpg` — `keep_core=0.05, keep_halo=0.3` 的三列对比子集
- `contact_sheet.jpg` — 全部 144 个 suppressed 缩略图
- `black_line_score_preview.jpg` — black-line score 全图预览（高亮处是黑线候选）
- `summary.csv` — 每个变体的 core/halo/outside 指标
- `shortlist.md` — Top 7 候选 + 选型说明
- `variants/<name>/{summary_panel,suppressed_preview,removed}.jpg` —
  Top variant 的核心图（共 7 × 3 = 21 张）

所有 JPG 最长边 ≤ 2800 px，quality=85。

## 重点查看文件

1. **`contact_sheet.jpg`** — 144 个变体 suppressed 缩略图整体扫一遍。
2. **`compare_sheet.jpg`** — `keep_core=0.05, keep_halo=0.3` 子集（24 行）的
   三列对比（`core+halo overlay` / `suppressed` / `raw - suppressed`）。
3. **`black_line_score_preview.jpg`** — 检查 black-line score 是否在黑线处
   亮、在亮脊处暗；这是 v2 mask 的关键输入。
4. **`shortlist.md`** + 各 `variants/<name>/summary_panel.jpg` — 7 个候选的
   raw / mask / suppressed / diff 四联图。
5. **`variants/<name>/suppressed_preview.jpg`** — 候选变体全分辨率单图，
   肉眼判断"主黑线是否还能看到"。
6. **`summary.csv`** 中重点列：
   - `core_coverage_percent`、`edit_coverage_percent`（编辑面积）
   - `mean_abs_removed_core`（核心区的实际抑制强度，v2 期望 > 200）
   - `mean_abs_removed_outside`（应趋近 0）

## Top variants

| # | variant_name | 选型场景 |
|---|---|---|
| 1 | `C90b_wc5_h8_kc0.00_kh0.3_s15`  | **主推**：窄核心 + 宽 halo + 核心完全替换 |
| 2 | `C90b_wc7_h8_kc0.00_kh0.3_s15`  | 处理较粗的线（核心 7px） |
| 3 | `C90b_wc5_h4_kc0.00_kh0.3_s15`  | 紧 halo，编辑区最小 |
| 4 | `C90b_wc5_h8_kc0.05_kh0.3_s15`  | 留 5% 核心残差，作对照 |
| 5 | `C90b_wc5_h8_kc0.00_kh0.2_s15`  | halo 也强压制（kh=0.2） |
| 6 | `D85b_wc5_h8_kc0.00_kh0.3_s15`  | D mask（更宽），覆盖更广的黑线 |
| 7 | `D85b_wc7_h8_kc0.00_kh0.3_s15`  | 最激进覆盖（edit_cov≈36.8%） |

详见 `shortlist.md`。

## 本轮明确不做

- ❌ 不做 defect detection
- ❌ 不训练任何模型（Route B U-Net / Route D inpainting U-Net 都不跑）
- ❌ 不做复杂 inpainting（PatchMatch、扩散模型、深度 inpainting 等）
- ❌ 不修改 `main` 分支；所有改动只在 worktree
  `experiment/visual-line-hide-v1`
- ❌ 不提交 `data/data.tif`、`outputs/`、任何 `.tif/.npz/.npy/.pt` 等大文件

v2 的全部任务是"接着 v1 的方向，把核心区真正替换掉，让黑线视觉上隐进背景"。
