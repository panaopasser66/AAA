# task_stage3.md — Kossel Line Candidate Review + Line Removal Baseline

## 0. 当前阶段判断

当前 Stage 1.3 + Stage 2 已经完成。现在不要继续死磕 `high_confidence_clean` 是否非空。

当前事实：

1. `high_confidence_clean = 0` 是合理结果，不是 bug。
2. `refined_static_line_mask` 已经比旧版好很多，static line 不再是主要瓶颈。
3. 主要瓶颈变成了 `dynamic_line_risk / refined_combined_line_exclusion` 太密。
4. axis1_angle 确实让 Kossel line 变化更明显，但直接把 axis1 加入 median fusion 后，不能得到理想 clean image。
5. 现在最有价值的不是 `high_confidence_clean`，而是：
   - `line_adjacent_uncertain_v2`
   - `texture_uncertain_v2`
   - 这些候选的多帧 crop 证据

所以 Stage 3 的目标不是继续降低阈值凑 high-confidence，而是：

> 建立人工复核优先级 + 生成第一批标注数据 + 尝试窄线 mask 的 line-removal/inpainting baseline。

---

## 1. Stage 3 总目标

请在当前工程基础上继续开发，不要重写工程。

Stage 3 分成两个部分：

### Stage 3A：Manual Review Ranking

目标：

1. 从 `line_adjacent_uncertain_v2` 和 `texture_uncertain_v2` 中重新排序候选。
2. 不是按“离线远”排序，而是按“最值得人工复核为 defect”的概率排序。
3. 输出 top manual-review candidate crops。
4. 生成一个人工标注 CSV 模板，为后续深度学习准备训练集。

### Stage 3B：Line Removal / Kossel-line PS Baseline

目标：

1. 使用 refined line skeleton 生成窄线 mask。
2. 不再使用巨大 exclusion mask 做硬排除。
3. 尝试多种传统 inpainting / local background replacement 方法。
4. 输出多种 cleaned image variants。
5. 在 cleaned image 上重新跑 5–7 px blob detector。
6. 输出 cleaned 后的候选 crop，供人工比较。

---

## 2. 不要做什么

本阶段不要做以下事情：

1. 不要训练深度学习模型。
2. 不要把 `line_adjacent_uncertain` 直接当 defect 标签。
3. 不要为了让 `high_confidence_clean` 非空而降低正式标准。
4. 不要用大面积 `combined_line_exclusion_mask_dilate12` 做 inpainting。
5. 不要把 axis1_angle 直接当作 clean median fusion 的主来源。
6. 不要覆盖 Stage 1.3 / Stage 2 已有输出。

---

# Part A — Manual Review Ranking

## 3. 输入

使用现有输出：

```text
outputs/candidates/candidates_line_adjacent_uncertain_v2.csv
outputs/candidates/candidates_texture_uncertain_v2.csv
outputs/candidate_crops/line_adjacent_uncertain_v2/
outputs/candidate_crops/texture_uncertain_v2/
outputs/fusion/combined_fused_delined.png
outputs/fusion/combined_fused_delined_v2.png
outputs/line_risk/combined_line_risk_v2.png
outputs/line_risk/refined_static_line_mask.png
outputs/line_risk/refined_combined_line_exclusion_dilate8.png
outputs/invalid_region/refined_safe_search_mask_dilate8.png
```

如果其中某些文件不存在，请在报告里说明，不要静默失败。

---

## 4. 新增 candidate manual-review score

新增模块：

```text
src/manual_review_ranking.py
```

请读取：

```text
candidates_line_adjacent_uncertain_v2.csv
candidates_texture_uncertain_v2.csv
```

合并成一个 manual-review pool。

每个候选新增以下评分字段：

```text
diameter_closeness_score
shape_score
persistence_score
uncovered_evidence_score
line_penalty_soft
texture_penalty_soft
review_priority_score
review_reason
```

建议评分逻辑：

### 4.1 diameter_closeness_score

目标直径暂定 5.5 px。

```python
diameter_closeness_score = exp(-abs(diameter_px - 5.5) / 2.0)
```

### 4.2 shape_score

偏好圆形、小面积、低长宽比。

建议：

```python
shape_score =
    circularity_score
    * aspect_ratio_score
    * anisotropy_score
```

其中：

```text
circularity_score:
  circularity 在 0.5~1.4 较好
  NaN 或极端值直接降权

aspect_ratio_score:
  aspect_ratio <= 1.5 最好
  1.5~2.5 中等
  >2.5 降权

anisotropy_score:
  anisotropy <= 2.0 最好
  2.0~3.5 中等
  >3.5 降权
```

### 4.3 persistence_score

偏好在多帧中重复出现。

可用：

```text
spot_present_when_uncovered
uncovered_frame_count
valid_frame_count
```

建议：

```python
persistence_score = spot_present_when_uncovered / max(uncovered_frame_count, 1)
```

但如果 `spot_present_when_uncovered < 2`，强降权。

### 4.4 line_penalty_soft

注意：不要因为靠线就直接归零。现在我们知道很多候选会靠 Kossel line。

改成软惩罚：

```text
distance_to_refined_static_line_skeleton 越大越好
distance_to_refined_combined_line_exclusion 越大越好
mean_line_risk_in_patch 越低越好
```

但如果候选尺寸和多帧证据很强，可以仍进入人工复核。

### 4.5 review_priority_score

建议综合：

```python
review_priority_score =
    0.30 * diameter_closeness_score
  + 0.25 * shape_score
  + 0.30 * persistence_score
  - 0.10 * line_penalty_soft
  - 0.05 * texture_penalty_soft
```

要求把公式写在代码注释和 report 里，便于后续调整。

---

## 5. Manual review 输出

新增脚本：

```text
scripts/06_rank_manual_review_candidates.py
```

运行后输出：

```text
outputs/manual_review/manual_review_candidates_all.csv
outputs/manual_review/manual_review_top50.csv
outputs/manual_review/manual_review_top100.csv
outputs/manual_review/manual_review_overlay_top50.png
outputs/manual_review/manual_review_overlay_top100.png
outputs/manual_review/crops_top50/
outputs/manual_review/crops_top100/
outputs/manual_review/review_index.html
outputs/manual_review/manual_labels_template.csv
```

### 5.1 manual_labels_template.csv

必须包含：

```text
candidate_id
source_category
x
y
diameter_px
polarity
review_priority_score
crop_path
manual_label
manual_confidence
notes
```

`manual_label` 允许值建议写在 CSV 注释或 report 中：

```text
true_defect
suspected_defect
kossel_line_artifact
texture_artifact
noise
edge_or_border
uncertain
```

`manual_confidence` 建议：

```text
1, 2, 3, 4, 5
```

---

## 6. Manual review crop sheet 增强

每个 top candidate crop sheet 应包含：

```text
0000 crop
axis2 aligned crops
axis4 aligned crops
axis1 aligned crops
combined_fused_delined crop
combined_fused_delined_v2 crop
combined_line_risk_v2 crop
refined_static_line_mask crop
refined_combined_line_exclusion crop
refined_safe_search_mask crop
```

同时在图像标题或右侧文本中显示：

```text
candidate_id
x, y
diameter_px
polarity
source_category
review_priority_score
diameter_closeness_score
shape_score
persistence_score
spot_present_when_uncovered
uncovered_frame_count
distance_to_refined_static_line_skeleton
distance_to_refined_combined_line_exclusion
anisotropy
aspect_ratio
circularity
```

Crop size 建议：

```yaml
manual_review:
  crop_size: 96
  top_k: [50, 100]
```

---

# Part B — Line Removal / Kossel-line PS Baseline

## 7. 目标

生成几种“去 Kossel line”的预览图，用于肉眼看 defect，也用于后续深度学习 pseudo-clean target。

注意：本阶段不是追求完美，而是比较不同 line-removal 方法。

---

## 8. 窄线 mask 生成

新增模块：

```text
src/line_removal_masks.py
```

不要使用大面积 exclusion mask。

使用以下作为基础：

```text
refined_static_line_skeleton
dynamic_line_skeleton
```

生成窄线 mask：

```text
line_mask_radius2
line_mask_radius4
line_mask_radius6
```

输出：

```text
outputs/line_removal/masks/line_mask_radius2.png
outputs/line_removal/masks/line_mask_radius4.png
outputs/line_removal/masks/line_mask_radius6.png
```

每个 mask 要统计 coverage：

```text
outputs/line_removal/masks/line_removal_mask_coverage.csv
```

建议 coverage：

```text
radius2: 低覆盖，用于最保守去线
radius4: 中等
radius6: 较激进
```

如果 coverage 超过 35%，在报告里标记 too_aggressive_for_inpainting。

---

## 9. Defect protect mask

为了避免把可能的 defect 一起 inpaint 掉，需要生成 protect mask。

来源：

```text
manual_review_top100 candidates
line_adjacent_uncertain_v2
texture_uncertain_v2
```

在每个 candidate 中心画小圆：

```text
radius = max(4, diameter_px / 2 + 2)
```

输出：

```text
outputs/line_removal/masks/defect_protect_mask.png
```

line-removal/inpainting 时提供两种版本：

```text
without_protect
with_protect
```

`with_protect` 版本中，candidate 中心区域不参与 inpainting。

---

## 10. Inpainting / replacement 方法

新增模块：

```text
src/line_inpainting.py
```

新增脚本：

```text
scripts/07_line_removal_baseline.py
```

对以下输入图分别测试：

```text
0000 normalized image
combined_fused_delined
combined_fused_delined_v2
```

方法至少包括：

### 10.1 local_median_replace

对 mask 内像素，用 mask 外邻域局部 median 替换。

参数：

```yaml
local_median:
  window_sizes: [15, 25, 41]
```

### 10.2 cv2_telea_inpaint

OpenCV Telea inpainting。

参数：

```yaml
telea:
  radius: [3, 5, 7]
```

### 10.3 cv2_ns_inpaint

OpenCV Navier-Stokes inpainting。

参数：

```yaml
navier_stokes:
  radius: [3, 5, 7]
```

### 10.4 directional_background_replace

可选，但建议实现简单版本：

1. 对局部 patch 估计 line direction。
2. 沿垂直于 line 的方向采样两侧背景。
3. 用两侧背景插值替换 line mask 内像素。

如果实现复杂，可以先跳过，但要在 report 中说明。

---

## 11. 输出 line removal variants

输出目录：

```text
outputs/line_removal/cleaned_images/
```

命名格式：

```text
cleaned_{input_name}_{mask_radius}_{method}_{param}_{protect_or_no_protect}.png
cleaned_{input_name}_{mask_radius}_{method}_{param}_{protect_or_no_protect}.npy
```

例如：

```text
cleaned_0000_r4_telea_rad5_protect.png
cleaned_fused_v1_r4_localmedian_w25_no_protect.png
```

同时输出对比图：

```text
outputs/line_removal/comparison_sheets/
```

每张 comparison sheet 包含：

```text
original
line_mask
cleaned
removed_residual = original - cleaned
zoom crops around selected candidate positions
```

---

## 12. 在 cleaned image 上重新检测 blob

新增脚本：

```text
scripts/08_detect_candidates_on_cleaned.py
```

对若干推荐 cleaned image 重新跑 blob detector。

第一版可选择：

```text
cleaned_0000_r4_telea_rad5_protect
cleaned_0000_r4_localmedian_w25_protect
cleaned_fused_v1_r4_telea_rad5_protect
cleaned_fused_v1_r4_localmedian_w25_protect
```

输出：

```text
outputs/line_removal/candidates_on_cleaned/{variant_name}/candidates.csv
outputs/line_removal/candidates_on_cleaned/{variant_name}/overlay.png
outputs/line_removal/candidates_on_cleaned/{variant_name}/candidate_crops/
```

Scoring 不要复用过强的 combined exclusion gate。  
cleaned image 上的筛选应更关注：

```text
diameter 3–7 px
area 10–80
circularity 0.5–1.4
aspect_ratio <= 2.0
anisotropy <= 3.0
local contrast
是否落在 protect mask 附近
```

但仍要输出与 line mask 的距离作为参考，不要完全忽略。

---

## 13. 报告更新

更新：

```text
outputs/reports/report.html
```

新增 Stage 3 章节：

```text
Stage 3A Manual Review Ranking
Stage 3B Line Removal Baseline
```

展示：

1. manual review top50 overlay
2. top50 crop thumbnails
3. manual_labels_template.csv 链接
4. line removal mask coverage
5. 不同 inpainting 方法对比
6. cleaned image variants
7. removed residual
8. candidates on cleaned images

---

## 14. config.yaml 新增参数

新增：

```yaml
manual_review:
  enabled: true
  crop_size: 96
  top_k: [50, 100]
  target_diameter_px: 5.5
  source_categories:
    - line_adjacent_uncertain_v2
    - texture_uncertain_v2

line_removal:
  enabled: true
  input_images:
    - "0000"
    - "combined_fused_delined"
    - "combined_fused_delined_v2"
  mask_radii: [2, 4, 6]
  protect_candidates: true
  protect_radius_extra_px: 2
  methods:
    local_median:
      enabled: true
      window_sizes: [15, 25, 41]
    telea:
      enabled: true
      radii: [3, 5, 7]
    navier_stokes:
      enabled: true
      radii: [3, 5, 7]

cleaned_candidate_detection:
  enabled: true
  selected_variants:
    - "cleaned_0000_r4_telea_rad5_protect"
    - "cleaned_0000_r4_localmedian_w25_protect"
    - "cleaned_fused_v1_r4_telea_rad5_protect"
    - "cleaned_fused_v1_r4_localmedian_w25_protect"
  min_diameter_px: 3
  max_diameter_px: 7
  target_diameter_px: 5.5
  min_area_px: 10
  max_area_px: 80
  min_circularity: 0.5
  max_circularity: 1.4
  max_aspect_ratio: 2.0
  max_anisotropy: 3.0
  max_candidates_per_variant: 300
```

---

## 15. 新的一键运行命令

请更新总入口脚本，让以下命令能完整跑 Stage 1–3：

```bash
python scripts/run_stage1.py --raw_dir raw --config configs/config.yaml --out_dir outputs
python scripts/06_rank_manual_review_candidates.py --config configs/config.yaml --out_dir outputs
python scripts/07_line_removal_baseline.py --raw_dir raw --config configs/config.yaml --out_dir outputs
python scripts/08_detect_candidates_on_cleaned.py --config configs/config.yaml --out_dir outputs
python scripts/05_make_report.py --config configs/config.yaml --out_dir outputs
```

如果可以，也新增：

```bash
python scripts/run_stage3.py --raw_dir raw --config configs/config.yaml --out_dir outputs
```

其中 `run_stage3.py` 只执行 Stage 3A 和 3B，不重复跑 Stage 1。

---

## 16. 验收标准

完成后请检查以下文件存在：

```text
outputs/manual_review/manual_review_top50.csv
outputs/manual_review/manual_review_overlay_top50.png
outputs/manual_review/crops_top50/
outputs/manual_review/review_index.html
outputs/manual_review/manual_labels_template.csv

outputs/line_removal/masks/line_mask_radius2.png
outputs/line_removal/masks/line_mask_radius4.png
outputs/line_removal/masks/line_mask_radius6.png
outputs/line_removal/masks/defect_protect_mask.png
outputs/line_removal/masks/line_removal_mask_coverage.csv

outputs/line_removal/cleaned_images/
outputs/line_removal/comparison_sheets/

outputs/line_removal/candidates_on_cleaned/
outputs/reports/report.html
```

报告中必须明确说明：

1. 当前仍未使用深度学习。
2. Stage 3A 是人工复核排序。
3. Stage 3B 是传统 inpainting / line-removal baseline。
4. Cleaned image 上检测出来的候选仍然不是最终 defect 标签，需要人工确认。
5. 这些人工确认标签将用于后续 Stage 4 深度学习。

---

## 17. 完成后请汇报

完成后请输出：

1. 新增/修改了哪些文件。
2. 一键运行命令。
3. manual review top50 数量。
4. line removal mask coverage。
5. 每种 cleaned variant 的候选数量。
6. 推荐人工优先查看的 cleaned variant。
7. 关键输出文件是否全部存在。
