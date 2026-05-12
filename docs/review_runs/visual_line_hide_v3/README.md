# Visual Line Hide v3 — Review Run

## 本轮目标

v2 review 结论：v2 比 v1 明显更对，black-line score / core+halo / core 硬替换
都对方向；**但 suppressed 图里贯穿主黑线仍然明显**，尤其在贯穿长线、星芒交叉、
右侧和下方的黑线。

v3 的目标：**实现“方向性线带背景替换”**，更接近 Photoshop 修线的做法。

### v3 的核心新动作：orientation-aware background filling

1. 用 **structure tensor** 估每个像素的局部 ridge 法向 `θ`：
   - 在 `gaussian(raw, σ=1)` 上求梯度 `gx, gy`
   - 平滑得到 `Jxx, Jxy, Jyy`（σ=4）
   - 主特征向量方向即是 ridge 的**法向**
2. 对每个 core / halo 像素，**沿法向 ± 两侧采样背景**：
   - 距离 `d ∈ {wc+2, wc+4, wc+8, wc+12}`
   - 8 个采样点（4 距离 × 2 方向）
   - 落在 `core | halo | broad_line_mask(sato75) | protect_mask` 上的样本作废
3. **per-pixel median** 有效样本即为 `bg_dir`
4. 没有任何有效采样的像素回退到 v2 的 **side-band gaussian bg**

### v3 还保留的 v2 设计

- `black_line_score = (bg_low - raw)+ / local_noise`，做 mask 过滤
- core + halo 双层 mask
- core 硬替换（`keep_core = 0.0`），halo 轻压制（`keep_halo` 可选）
- core 在 feather 之后**再次硬覆盖**

### v3 新增的可调维度

- `black_threshold` 不再固定 1.0，扫 **0.5 / 0.8 / 1.0**

本轮仍然**不做** defect detection，**不训练模型**，**不做复杂 inpainting**。

## 运行命令

```bash
# prepare 阶段产物可复用（如还在）
python scripts/route_d_selfsupervised_inpainting/prepare_project2_data.py \
    --input data/data.tif --out outputs/prepare

# v3 sweep（32 个变体，~13 分钟）
python scripts/route_c_bg_sampling_band_suppress/line_band_contrast_suppress_v3.py \
    --input data/data.tif \
    --prepared outputs/prepare \
    --out outputs/visual_line_hide_v3
```

### Sweep 配置（精简到 32）

| 维度 | C90b 取值 | D85b 取值 |
|---|---|---|
| mask | `C_sato90` ∩ black | `D_sato85` ∩ black |
| black_threshold | 0.5 / 0.8 / 1.0 | 0.5 / 1.0 |
| core_width | 5 / 7 | 5 / 7 |
| halo_width | 6 / 8 | 8 |
| keep_halo | 0.2 / 0.3 | 0.2 / 0.3 |
| keep_core | 0.0（固定） | 0.0（固定） |
| side_sigma | 15（固定） | 15（固定） |
| tensor_sigma | 4（固定） | 4（固定） |

合计 24 + 8 = **32 个变体**。D85b 仅作为激进对照。

## 输入路径

- 原始图：`data/data.tif`（被 `.gitignore` 忽略，不进 Git）
- prepare 输出：`outputs/prepare/`（被 `.gitignore` 忽略，不进 Git）
  - `masks/mask_C_sato90.png`、`masks/mask_D_sato85.png`
  - `masks/mask_F_sato75.png` —— 用作 `broad_line_mask`，禁止从邻线取背景
  - `masks/protect_mask.png`
  - `responses/response_sato.tif` —— 用于自动选 4 个 zoom 中心

## 输出路径

### 完整产物（不入 Git）

- `outputs/visual_line_hide_v3/black_line_score_preview.png`
- `outputs/visual_line_hide_v3/coherence_preview.png`
- `outputs/visual_line_hide_v3/reports/`
  - `contact_sheet.png`、`compare_sheet.png`、`compare_sheet_all.png`
  - `zoom_compare_sheet.png`
  - `summary.csv`、`black_line_score_stats.json`、`zoom_crops.json`
- `outputs/visual_line_hide_v3/variants/<name>/`
  - `summary_panel.png`（含 4 个 zoom crop 在第二行）
  - `suppressed_preview.png`、`removed.png`、`overlay.png`
  - `core_mask.png`、`halo_mask.png`
  - `suppressed.tif`、`background_estimate.tif`、`metrics.json`

### 审核可提交产物（本目录，已入 Git）

- `compare_sheet.jpg` —— `keep_halo=0.3` 子集的三列对比
- `contact_sheet.jpg` —— 全部 32 个变体的 suppressed 缩略图
- **`zoom_compare_sheet.jpg`** —— Top 5 variants × 4 zoom 区域 vs raw，
  专门给 v2 review 中"看不清细节"的痛点
- `black_line_score_preview.jpg` —— v3 沿用 v2 的 black-line score 可视化
- `coherence_preview.jpg` —— structure tensor 的 coherence 图，亮处方向性强，
  暗处方向估计不可靠（用于诊断方向性 bg 是否落到非线区域）
- `summary.csv`、`shortlist.md`
- `variants/<top>/{summary_panel,suppressed_preview,removed}.jpg` × 7

所有 JPG 最长边 ≤ 2800 px，quality=85。

## 重点查看文件

1. **`zoom_compare_sheet.jpg`** —— **本轮最关键的一张**。Top 5 variants
   在 4 个 zoom 区域上的 suppressed crop，与 raw 同列对比；上面 v2 看不清
   的细节在 480×480 crop 上一目了然。
2. **`contact_sheet.jpg`** —— 32 个变体 suppressed 全图缩略，整体扫一遍。
3. **`shortlist.md`** + `variants/<top>/summary_panel.jpg` —— 每个 panel
   现在有两行：第一行是 raw/overlay/suppressed/removed 全图，第二行是
   4 个 zoom 在 suppressed 下的 crop。一张 panel 同时看全局和细节。
4. **`variants/<top>/suppressed_preview.jpg`** —— 候选变体全分辨率单图，
   肉眼判断"主黑线、星芒交叉、右下黑线"是否已经基本看不见。
5. **`summary.csv` 的 `core_residual_after_suppress` 列** ——
   越接近 0，黑线越彻底融入背景（v3 主推变体已经做到 +0.5）。

## Top variants

| # | variant_name | resid | dir% | 备注 |
|---|---|---:|---:|---|
| 1 | `C90b_bt1.0_wc7_h6_kh0.3` | **+0.50** | 59.3 | **主推**：核心残留几乎为 0 |
| 2 | `C90b_bt1.0_wc7_h6_kh0.2` | **+0.50** | 59.3 | 同上，halo 也更狠（kh=0.2） |
| 3 | `C90b_bt0.8_wc5_h6_kh0.3` | -9.96 | 53.6 | 较小核心，bt 中等 |
| 4 | `C90b_bt0.8_wc7_h6_kh0.3` | -20.35 | 55.6 | 较宽核心，bt 中等 |
| 5 | `C90b_bt0.5_wc7_h8_kh0.3` | -54.01 | 38.0 | bt 最低（覆盖最多线），edit% 较大 |
| 6 | `D85b_bt0.5_wc7_h8_kh0.3` | -53.69 | 30.9 | D85 + bt 最低，最激进 |
| 7 | `D85b_bt1.0_wc7_h8_kh0.3` | -40.51 | 43.2 | D85 + bt 严格 |

详见 `shortlist.md`。

## 本轮明确不做

- ❌ 不做 defect detection
- ❌ 不训练任何模型（Route B U-Net / Route D inpainting U-Net 都不跑）
- ❌ 不做复杂 inpainting（PatchMatch、扩散模型、深度 inpainting 等）
- ❌ 不修改 `main` 分支；所有改动只在 worktree
  `experiment/visual-line-hide-v1`
- ❌ 不提交 `data/data.tif`、`outputs/`、任何 `.tif/.npz/.npy/.pt` 等大文件

v3 全部任务是"用方向性背景采样把核心残留压到 0 附近"。
