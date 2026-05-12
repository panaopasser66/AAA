# Visual Line Hide v5 (deepmask) — Review Run

## 本轮目标

v4 review 结论：v4 是目前最好的**规则版**，但没达到最终目标——能清掉部分断续
黑点/黑段，但仍有明显线残留。**不要再单纯叠 v5 规则。**

v5 切换思路：

> **v5 = deep line mask + v4 orientation-aware suppression**

把 mask 来源从纯 sato+black-score 规则换成**通用 line probability map**（未来
由 Route B U-Net 提供），整个 orientation-aware 的 bg 估计和 core/halo
管线（来自 v3/v4）保持不变。

本轮没有训练模型，所以 line_prob 用 **合成 fallback** 暂代：
`prob = sigmoid(6 · sato_norm · black_score − 2)`，平滑 σ=0.8。
代码接口已准备好，**真正的 U-Net 出来后只需 `--line-prob path/to/prob.tif`**。

### v5 不变的部分（v4 carryover）

- 结构张量估 ridge 法向
- 沿法向 ±d 在 {wc+2, wc+4, wc+8, wc+12} 采样、median reduce
- side-band gaussian bg 作 fallback
- protect mask 分 `true_blob` / `on_line`（48 + 394）
- core 硬替换、halo residual attenuation、feather + 二次硬覆盖

### v5 唯一新的部分

- Mask 来源接口三选一：
  1. `--line-prob path/to/prob.tif`（首选，未来 U-Net 输出）
  2. `--line-mask path/to/mask.png`（任意已生成的二值线 mask）
  3. fallback 合成（本轮使用）

本轮仍然**不做** defect detection / 模型训练 / self-supervised inpainting。

## 运行命令

```bash
# 当前：合成 prob fallback
python scripts/route_c_bg_sampling_band_suppress/line_band_contrast_suppress_v5_deepmask.py \
    --input data/data.tif \
    --prepared outputs/prepare \
    --out outputs/visual_line_hide_v5_deepmask

# 未来：用 Route B U-Net 推理结果
python scripts/route_c_bg_sampling_band_suppress/line_band_contrast_suppress_v5_deepmask.py \
    --input data/data.tif \
    --prepared outputs/prepare \
    --line-prob outputs/route_b/unet_line_prob.tif \
    --out outputs/visual_line_hide_v5_unet
```

### Sweep 配置（精简到 10）

5 对 (core_threshold, halo_threshold) × 2 个 closing_len：

| pair | core_th | halo_th |
|---:|---:|---:|
| 1 | 0.50 | 0.25 |
| 2 | 0.50 | 0.35 |
| 3 | 0.60 | 0.35 |
| 4 | 0.60 | 0.45 |
| 5 | 0.70 | 0.45 |

× closing_len ∈ {0, 15}。固定：closing_width=1, halo_dilate=2, keep_core=0.0,
keep_halo=0.3, tensor_sigma=4, distances={9,11,15,19}。

合计 **10 个变体**，~5 分钟跑完（295s）。

## 输入路径

- `data/data.tif`
- `outputs/prepare/`
  - `responses/response_sato.tif`（用于合成 fallback + zoom crop 选择）
  - `masks/protect_mask.png`、`mask_F_sato75.png`

## 输出路径

### 完整产物（不入 Git）

- `outputs/visual_line_hide_v5_deepmask/line_prob.tif`、
  `line_prob_preview.png`、`protect_*.png`
- `outputs/visual_line_hide_v5_deepmask/reports/`
  - `summary.csv`、`contact_sheet.png`、`compare_sheet.png`、
    `zoom_compare_sheet.png`、`line_prob_info.json`、`zoom_crops.json`
- `outputs/visual_line_hide_v5_deepmask/variants/<name>/`
  - `summary_panel.png`（含 4 个 zoom crop）
  - `suppressed_preview.png`、`removed.png`、`overlay.png`、`cleanup_mask.png`
  - `core_mask.png`、`halo_mask.png`、`suppressed.tif`、`metrics.json`

### 审核可提交产物（本目录，已入 Git）

- `contact_sheet.jpg` — 10 个变体的 suppressed 缩略
- `compare_sheet.jpg` — 10 行 (core+halo / suppressed / removed)
- `zoom_compare_sheet.jpg` — Top 5 variants × 4 zoom + raw 行
- **`line_prob_preview.jpg`** — 合成 line_prob 的全图视觉
- `protect_true_blob.jpg` / `protect_on_line.jpg`（同 v4）
- `summary.csv`、`shortlist.md`、`line_prob_info.json`
- `variants/<top>/{summary_panel,suppressed_preview,removed,cleanup_mask}.jpg` × 7

所有 JPG 最长边 ≤ 2800 px，quality=85。

## 重点查看文件

1. **`line_prob_preview.jpg`** —— 第一时间看合成 prob 是否合理：
   - 暗线处应该亮（接近 1）
   - 背景应该暗（接近 0）
   - 当前合成 prob `p50=0.14, p99=0.57` —— 偏窄，瓶颈在这里
2. **`zoom_compare_sheet.jpg`** —— v5 各变体在 4 个 zoom 上的表现
3. **`shortlist.md`** —— Top 7 + v3/v4/v5 主推对比
4. **`variants/v5_ct0.50_ht0.35_cl0/summary_panel.jpg`** —— 主推变体的细节
5. `summary.csv` 中 `core_residual_after_suppress` 列：
   - v3 主推（重算 v4 bg 基准）+24
   - v4 主推 +18.7
   - **v5 主推 -0.60**  —— core_resid 最接近 0（line 几乎完全融入 bg_low）

## Top variants

| # | variant_name | core% | edit% | core_resid | 备注 |
|---|---|---:|---:|---:|---|
| 1 | `v5_ct0.50_ht0.35_cl0`  | 1.58 | 14.07 | **-0.60** | **主推** |
| 2 | `v5_ct0.50_ht0.35_cl15` | 3.95 | 14.77 | -5.02 | +closing |
| 3 | `v5_ct0.60_ht0.35_cl0`  | 0.79 | 14.07 | -7.32 | 紧 core |
| 4 | `v5_ct0.60_ht0.35_cl15` | 1.76 | 14.29 | -8.76 | 紧 core +closing |
| 5 | `v5_ct0.50_ht0.25_cl15` | 3.95 | 30.97 | -67.74 | **激进**（灰带风险） |
| 6 | `v5_ct0.60_ht0.45_cl15` | 1.76 |  7.67 | +34.82 | 紧 halo |
| 7 | `v5_ct0.70_ht0.45_cl15` | 0.70 |  7.43 | +35.64 | 最保守 |

详见 `shortlist.md`。

## 关键观察

1. **v5 主推 `core_resid = -0.60`** —— core 在抑制后几乎完全融入了 bg_low。
   v3/v4 的同位指标在 +18~+25 量级，v5 明显更接近"完美隐线"目标。
2. **但 v5 的 core_cov 只有 1.58%**，远窄于 v3/v4 的 8.02% ——
   **瓶颈在合成 line_prob 的窄度**：`prob > 0.5` 只覆盖了 1.58% 像素，
   远小于实际暗线宽度。
3. **结论**：v5 的 *架构* 正确，*pipeline* 工作良好，**最后一步靠 Route B 的
   U-Net 把 line_prob 拓宽到与真实暗线宽度相当**。当 prob > 0.5 能覆盖
   ~8% 像素（与 v3/v4 的 core_cov 相当）时，v5 应可显著超过 v4。
4. **下一步选项**：
   a. 训练 Route B U-Net（已在 scripts/route_b_line_mask_unet/），用 v3/v4
      的 sato+black mask 作伪标签自监督；
   b. 或先手动改 synthesis 的 logit_gain/bias，让合成 prob 在线带内饱和；
   c. 验证："prob → core → suppression" 全链路在更宽 prob 输入下是否仍稳。

## 本轮明确不做

- ❌ 不做 defect detection
- ❌ 不在本脚本里训练任何模型（脚本只**接入**线 prob，不**产生** prob）
- ❌ 不回到 self-supervised inpainting
- ❌ 不修改 `main` 分支；所有改动只在 worktree
  `experiment/visual-line-hide-v1`
- ❌ 不提交 `data/data.tif`、`outputs/`、`.tif/.npz/.npy/.pt` 等大文件

v5 的全部任务是 **把 mask 接口从规则切换到 deep prob，验证 architecture 走通**。
