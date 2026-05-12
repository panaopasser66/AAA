# task_stage1.md

# 多位姿 SiC X-ray Kossel line / defect 候选检测 Stage 1 工程任务

## 0. 任务背景

当前项目用于处理碳化硅晶锭 X-ray TIFF 图像。图像中存在大量 Kossel line，这些线会干扰肉眼观察和后续缺陷检测。用户已采集同一样品在多个位姿下的 TIFF 图像。

核心实验事实：

1. defect 属于样品本体结构。图像配准到样品坐标后，真实 defect 应该在多张图中相对稳定。
2. Kossel line 会随着样品台角度、左右移动、上下移动、倍率调整发生变化。图像配准到样品坐标后，Kossel line 不一定稳定。
3. 用户目前重点关注的微管类 defect 大小大约为直径 5–6 pixel，但后续可能扩展到不同尺寸 defect 的统计。
4. 第一阶段不要直接训练深度学习模型。第一阶段目标是建立一个稳定、可验证、可复现的多图证据系统，为后续 deep learning 生成 pseudo-clean image 和人工复核标签。

本任务只做 Stage 1：多帧配准、多图融合去线预览、Kossel line 风险图、5–10 px 圆形 defect 候选检测、候选 crop 证据导出、HTML 报告。

---

## 1. 当前输入数据

用户已经建立好 `raw/` 文件夹，里面有以下命名规则的 TIFF 文件。

示例：

```text
0000.tif
0001.tif
000-1.tif
0010.tif
00-10.tif
0100.tif
0-100.tif
1000.tif
-1000.tif
2000.tif
-2000.tif
```

文件名四位分别表示：

```text
第 1 位：轴1，角度/转动
第 2 位：轴2，左右移动
第 3 位：轴3，倍率/放大率调整
第 4 位：轴4，上下移动
```

其中：

```text
0  表示初始位置
1  表示正向移动一个单位
-1 表示反向移动一个单位
2  表示正向移动两个单位
-2 表示反向移动两个单位
```

例如：

```text
0000.tif   -> axis1=0,  axis2=0,  axis3=0,  axis4=0
0100.tif   -> axis1=0,  axis2=1,  axis3=0,  axis4=0
0-100.tif  -> axis1=0,  axis2=-1, axis3=0,  axis4=0
0001.tif   -> axis1=0,  axis2=0,  axis3=0,  axis4=1
000-1.tif  -> axis1=0,  axis2=0,  axis3=0,  axis4=-1
1000.tif   -> axis1=1,  axis2=0,  axis3=0,  axis4=0
-1000.tif  -> axis1=-1, axis2=0,  axis3=0,  axis4=0
2000.tif   -> axis1=2,  axis2=0,  axis3=0,  axis4=0
-2000.tif  -> axis1=-2, axis2=0,  axis3=0,  axis4=0
0010.tif   -> axis1=0,  axis2=0,  axis3=1,  axis4=0
00-10.tif  -> axis1=0,  axis2=0,  axis3=-1, axis4=0
```

### 关键要求：文件名解析

请实现一个稳健的解析函数。

推荐方式：

```python
import re
tokens = re.findall(r"-?\d", stem)
```

要求：

```text
stem = "0000"   -> ["0", "0", "0", "0"]
stem = "0-100"  -> ["0", "-1", "0", "0"]
stem = "000-1"  -> ["0", "0", "0", "-1"]
stem = "-1000"  -> ["-1", "0", "0", "0"]
stem = "-2000"  -> ["-2", "0", "0", "0"]
stem = "2000"   -> ["2", "0", "0", "0"]
```

必须断言恰好解析出 4 个轴值，否则报出清晰错误。

---

## 2. Stage 1 总体目标

建立一个 Python 工程，输入 `raw/*.tif`，输出：

```text
1. 数据检查结果 metadata.csv
2. 每张 TIFF 的预览图 preview PNG
3. 每张图归一化后的 npy/png
4. axis2_x 和 axis4_y 两组图配准到 0000.tif 后的 aligned image
5. 配准质量检查图 registration QC
6. 每组的多图融合去线预览图 fused_median
7. 每组的 Kossel line 变化风险图 line_risk / MAD
8. axis2 + axis4 合并后的 combined_fused_delined image
9. 5–10 px 小圆形 defect 候选检测 overlay
10. 每个候选点的多图 crop 证据图
11. candidates.csv
12. report.html
```

第一阶段默认只启用：

```text
axis2_x: 0-100.tif, 0000.tif, 0100.tif
axis4_y: 000-1.tif, 0000.tif, 0001.tif
```

暂时不要把 axis1_angle 和 axis3_mag 混进主流程。

原因：

```text
axis2_x / axis4_y 主要是平移，配准最简单，最适合验证“defect稳定、Kossel line变化”的假设。
axis1_angle 涉及旋转/投影变化，第二阶段再启用。
axis3_mag 涉及缩放/清晰度/插值变化，第三阶段再启用。
```

---

## 3. 工程目录结构

请按下面结构建立工程：

```text
project_multiframe_defect/
│
├── raw/
│   ├── 0000.tif
│   ├── 0001.tif
│   ├── 000-1.tif
│   ├── 0010.tif
│   ├── 00-10.tif
│   ├── 0100.tif
│   ├── 0-100.tif
│   ├── 1000.tif
│   ├── -1000.tif
│   ├── 2000.tif
│   └── -2000.tif
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── __init__.py
│   ├── io_utils.py
│   ├── filename_parser.py
│   ├── normalize.py
│   ├── groups.py
│   ├── registration.py
│   ├── fusion.py
│   ├── line_risk.py
│   ├── blob_detection.py
│   ├── candidate_scoring.py
│   ├── crop_export.py
│   └── report.py
│
├── scripts/
│   ├── 00_inspect_dataset.py
│   ├── 01_register_groups.py
│   ├── 02_fuse_delined.py
│   ├── 03_detect_candidates.py
│   ├── 04_export_candidate_crops.py
│   ├── 05_make_report.py
│   └── run_stage1.py
│
├── outputs/
│   ├── inspect/
│   ├── normalized/
│   ├── aligned/
│   ├── registration_qc/
│   ├── fusion/
│   ├── line_risk/
│   ├── candidates/
│   ├── candidate_crops/
│   └── reports/
│
├── requirements.txt
└── README.md
```

如果当前文件夹已经有 `raw/`，不要移动或修改原始 TIFF。

---

## 4. 依赖要求

请创建 `requirements.txt`：

```text
numpy
scipy
pandas
tifffile
opencv-python
scikit-image
matplotlib
pyyaml
jinja2
tqdm
```

所有脚本都要能从项目根目录运行。

---

## 5. 配置文件要求

创建 `configs/config.yaml`：

```yaml
reference_image: "0000.tif"

groups:
  axis2_x:
    files: ["0-100.tif", "0000.tif", "0100.tif"]
    transform_model: "translation"
    enabled: true

  axis4_y:
    files: ["000-1.tif", "0000.tif", "0001.tif"]
    transform_model: "translation"
    enabled: true

  axis1_angle:
    files: ["-2000.tif", "-1000.tif", "0000.tif", "1000.tif", "2000.tif"]
    transform_model: "similarity"
    enabled: false

  axis3_mag:
    files: ["00-10.tif", "0000.tif", "0010.tif"]
    transform_model: "similarity"
    enabled: false

normalization:
  percentile_low: 0.5
  percentile_high: 99.5
  output_dtype: "float32"
  match_reference_median_mad: true
  preview_percentile_low: 1.0
  preview_percentile_high: 99.0

registration:
  registration_blur_sigma: 5.0
  use_phase_correlation_init: true
  use_ecc_refine: true
  ecc_max_iterations: 300
  ecc_eps: 1.0e-6
  allow_ecc_failure_fallback: true
  max_abs_shift_px: 200

fusion:
  method: "median"
  compute_mad: true
  compute_std: true
  combined_method: "median"

line_risk:
  normalize_percentile_high: 99.5
  high_risk_percentile: 95.0

blob_detection:
  detect_bright: true
  detect_dark: true
  min_diameter_px: 3
  max_diameter_px: 10
  target_diameter_px: 6
  log_sigma_min: 1.0
  log_sigma_max: 3.0
  num_sigma: 8
  threshold_abs: null
  threshold_rel: 0.04
  overlap: 0.5
  exclude_border_px: 32

candidate_filter:
  min_area_px: 7
  max_area_px: 160
  min_circularity: 0.25
  max_aspect_ratio: 4.0
  max_candidates: 500
  nms_distance_px: 5

candidate_scoring:
  crop_radius_for_contrast_px: 5
  annulus_inner_radius_px: 8
  annulus_outer_radius_px: 14
  persistence_contrast_threshold: 0.02
  line_risk_penalty_weight: 0.35

crop_export:
  crop_size: 80
  draw_candidate_circle: true
  include_all_aligned_frames: true
  include_fusion: true
  include_line_risk: true
  include_overlay: true
  max_crops: 300

report:
  max_candidates_in_html: 300
```

---

## 6. 模块详细要求

### 6.1 `src/io_utils.py`

需要实现：

```python
def load_tif_float32(path) -> np.ndarray:
    """Load tif as float32 2D image. Preserve raw values before normalization."""
```

要求：

```text
1. 支持 uint16/float32/float64 等常见 TIFF dtype。
2. 如果 TIFF 是 3D 或多页，第一版先报错，不要悄悄取第一帧。
3. 返回 float32 copy。
4. 不修改原始文件。
```

还需要实现：

```python
def robust_rescale_for_preview(img, p_low=1.0, p_high=99.0) -> np.ndarray:
    """Return uint8 preview image in [0,255]."""
```

```python
def save_png(path, img_uint8_or_float) -> None:
    """Save image to PNG."""
```

```python
def ensure_dir(path) -> None:
    """Create directory if needed."""
```

---

### 6.2 `src/filename_parser.py`

需要实现：

```python
@dataclass
class MotionInfo:
    filename: str
    stem: str
    axis1: int
    axis2: int
    axis3: int
    axis4: int
```

```python
def parse_motion_filename(filename: str) -> MotionInfo:
    """Parse names like 0000.tif, 0-100.tif, 000-1.tif, -2000.tif."""
```

```python
def scan_raw_dir(raw_dir: str) -> pd.DataFrame:
    """Return metadata dataframe with filename, path, axis1..axis4."""
```

必须测试这些 case：

```text
0000.tif  -> 0,0,0,0
0100.tif  -> 0,1,0,0
0-100.tif -> 0,-1,0,0
0001.tif  -> 0,0,0,1
000-1.tif -> 0,0,0,-1
1000.tif  -> 1,0,0,0
-1000.tif -> -1,0,0,0
2000.tif  -> 2,0,0,0
-2000.tif -> -2,0,0,0
0010.tif  -> 0,0,1,0
00-10.tif -> 0,0,-1,0
```

---

### 6.3 `src/normalize.py`

需要实现：

```python
def percentile_normalize(img, low=0.5, high=99.5) -> np.ndarray:
    """Robust percentile normalization to float32 [0,1]."""
```

```python
def match_median_mad_to_reference(img, ref, eps=1e-6) -> np.ndarray:
    """Match image median and MAD to reference image after percentile normalization."""
```

要求：

```text
1. 输出 float32。
2. 严格处理 NaN/Inf。
3. high <= low 时要 fallback 到 min/max 或报清晰错误。
4. 不要使用对比度增强 PNG 做计算，只能从 raw TIFF 读取。
```

---

### 6.4 `src/groups.py`

需要实现：

```python
def load_config(config_path: str) -> dict:
    """Load YAML config."""
```

```python
def get_enabled_groups(config: dict) -> dict:
    """Return enabled groups only."""
```

```python
def validate_group_files(raw_dir: str, config: dict) -> None:
    """Check all configured files exist. Raise clear error if missing."""
```

---

### 6.5 `src/registration.py`

第一版重点做 translation registration。

需要实现：

```python
def build_registration_image(img, blur_sigma=5.0) -> np.ndarray:
    """
    Build a smoother image for registration.
    Use Gaussian blur to reduce small defect/noise influence.
    Return float32 image.
    """
```

```python
def estimate_translation_phase(ref_reg, moving_reg) -> tuple[float, float]:
    """
    Estimate shift from moving to ref using phase correlation.
    Return dx, dy in pixels.
    """
```

```python
def refine_translation_ecc(ref_reg, moving_reg, init_dx, init_dy, max_iterations, eps):
    """
    Use OpenCV ECC to refine translation.
    If ECC fails, caller can fall back to phase correlation result.
    Return dx, dy, ecc_score, success.
    """
```

```python
def warp_translation(img, dx, dy, output_shape) -> np.ndarray:
    """
    Warp moving image into reference coordinate.
    Use cv2.warpAffine.
    Border mode should be reflect or constant NaN/0, but document choice.
    """
```

```python
def register_group(group_name, group_files, raw_dir, normalized_dir, out_dir, config) -> pd.DataFrame:
    """
    Register all images in group to reference 0000.tif.
    Save aligned .npy and preview .png.
    Save transform CSV.
    Save QC images.
    """
```

要求：

```text
1. 0000.tif 的 transform 必须是 dx=0, dy=0。
2. 对 axis2_x 和 axis4_y 使用 translation。
3. 如果 ECC 失败，必须 fallback 到 phase correlation，不允许直接中断全流程。
4. 如果平移量超过 config 中 max_abs_shift_px，给出 warning，并记录到 transforms.csv。
5. 保存 aligned float32 npy。
6. 保存 aligned preview png。
```

配准 QC 输出至少包括：

```text
outputs/registration_qc/{group_name}_transforms.csv
outputs/registration_qc/{group_name}_before_after_diff_contact.png
outputs/registration_qc/{group_name}_aligned_contact_sheet.png
```

---

### 6.6 `src/fusion.py`

需要实现：

```python
def load_aligned_stack(aligned_dir, group_name, filenames) -> tuple[np.ndarray, list[str]]:
    """Load aligned npy stack with shape [N,H,W]."""
```

```python
def fuse_stack(stack) -> dict:
    """
    Return:
    median
    mean
    std
    mad
    """
```

```python
def combine_group_fusions(group_fused_images: list[np.ndarray], method="median") -> np.ndarray:
    """Combine axis2 and axis4 fused images."""
```

输出：

```text
outputs/fusion/{group_name}_fused_median.npy
outputs/fusion/{group_name}_fused_median.png
outputs/fusion/{group_name}_fused_mean.png
outputs/fusion/combined_fused_delined.npy
outputs/fusion/combined_fused_delined.png
```

说明：

```text
median_image 是第一版的去 Kossel line 预览图。
由于只有 3 张一组，不保证完全去线，但应该能减弱随位姿变化的线结构。
```

---

### 6.7 `src/line_risk.py`

需要实现：

```python
def compute_line_risk_from_mad(mad_img, percentile_high=99.5) -> np.ndarray:
    """Normalize MAD/std image into [0,1] line-risk map."""
```

```python
def combine_line_risks(risk_maps: list[np.ndarray]) -> np.ndarray:
    """Use max or robust max to combine group risks."""
```

输出：

```text
outputs/line_risk/{group_name}_mad.npy
outputs/line_risk/{group_name}_mad.png
outputs/line_risk/{group_name}_line_risk.npy
outputs/line_risk/{group_name}_line_risk.png
outputs/line_risk/combined_line_risk.npy
outputs/line_risk/combined_line_risk.png
```

解释：

```text
line_risk 高的位置表示多张对齐图中变化大，很可能与 Kossel line、配准误差或强局部变化有关。
不要直接删除 line_risk 高区域的 defect 候选，只在评分中降权，并在 crop 报告里给用户人工判断。
```

---

### 6.8 `src/blob_detection.py`

第一版重点检测 3–10 px 直径的亮/暗圆形 blob。

推荐使用：

```python
skimage.feature.blob_log
```

需要实现：

```python
@dataclass
class Candidate:
    candidate_id: int
    x: float
    y: float
    radius: float
    diameter_px: float
    polarity: str  # "bright" or "dark"
    log_response: float | None
```

```python
def detect_blob_candidates(img, config) -> list[Candidate]:
    """
    Detect bright and dark blob candidates on combined_fused_delined image.
    Use LoG/DoG style detection.
    """
```

注意：

```text
1. bright blob：直接对 img 检测。
2. dark blob：对 -img 或 1-img 检测。
3. 检测前可以轻微 Gaussian smooth，但不要把 5–6 px defect 抹掉。
4. exclude_border_px 内的候选先排除，避免边界伪影。
5. 最后做 NMS，避免一个 defect 输出多个候选。
```

---

### 6.9 `src/candidate_scoring.py`

需要实现局部统计，不要求第一版特别复杂，但必须可解释。

建议实现：

```python
def local_contrast_score(img, x, y, r_center, r_inner, r_outer) -> float:
    """
    Compare center disk intensity with surrounding annulus.
    For bright defect: center - annulus median.
    For dark defect: annulus median - center.
    """
```

```python
def circularity_and_aspect_from_patch(img, x, y, radius, polarity) -> dict:
    """
    Estimate local connected component around candidate after local threshold.
    Return area, circularity, aspect_ratio.
    If failed, return NaN but do not crash.
    """
```

```python
def persistence_score(aligned_images, x, y, polarity, config) -> dict:
    """
    Check whether local contrast repeats across aligned frames.
    Return persistence_count, persistence_ratio, per_frame_contrasts.
    """
```

```python
def score_candidates(candidates, fused_img, line_risk, aligned_images, config) -> pd.DataFrame:
    """
    Build candidates.csv dataframe with interpretable columns.
    """
```

`candidates.csv` 至少包含：

```text
candidate_id
x
y
radius_px
diameter_px
polarity
area_px
circularity
aspect_ratio
local_contrast
log_response
persistence_count
persistence_ratio
line_risk_score
combined_score
class_suggestion
crop_path
```

评分建议：

```text
combined_score =
  + normalized local contrast
  + persistence_ratio
  + circularity bonus
  - line_risk_penalty_weight * line_risk_score
```

注意：

```text
line_risk_score 高不能直接删除候选。
真实 defect 可能刚好被 Kossel line 盖住。
```

分类建议：

```text
small_round_candidate: diameter 3–10 px, circularity 合理
medium_blob_candidate: diameter 8–20 px，第一版可先少量保留
line_like_artifact: aspect_ratio 大、circularity 低、persistence 差
uncertain: 其他情况
```

---

### 6.10 `src/crop_export.py`

这是最重要的人工复核输出之一。

需要实现：

```python
def export_candidate_crop_sheet(candidate_row, images_dict, out_path, config) -> str:
    """
    Export one contact sheet for a candidate.
    Include all aligned frames, fused crop, line risk crop, overlay crop.
    Draw candidate circle at center.
    """
```

每个候选点输出一张 PNG，例如：

```text
outputs/candidate_crops/candidate_0001_x1234_y0567_score0.821.png
```

crop sheet 至少包含：

```text
0000 aligned crop
axis2_x / 0-100 aligned crop
axis2_x / 0100 aligned crop
axis4_y / 000-1 aligned crop
axis4_y / 0001 aligned crop
combined_fused crop
combined_line_risk crop
overlay crop
```

显示要求：

```text
1. 每个 crop 用同样大小，比如 80x80。
2. 每个 crop 标题写清楚来自哪张图。
3. 候选位置画圆圈。
4. 不要对不同帧用完全不同的拉伸方式导致肉眼误判。可以用统一 percentile 或 crop 局部统一尺度，但必须在 README 中说明。
```

---

### 6.11 `src/report.py`

需要生成：

```text
outputs/reports/report.html
```

报告内容包括：

```text
1. 数据概况：多少张图、shape、dtype、灰度范围。
2. 文件名解析表。
3. 启用的 group。
4. 每组配准 transform 表。
5. 融合图和 line risk 图预览。
6. candidate overlay。
7. candidates.csv 下载/路径提示。
8. 候选点表格：candidate_id, x, y, diameter, polarity, score, persistence, line_risk, crop image。
```

HTML 不要求漂亮，但要方便快速浏览 candidate crop。

---

## 7. 脚本详细要求

### 7.1 `scripts/00_inspect_dataset.py`

命令：

```bash
python scripts/00_inspect_dataset.py --raw_dir raw --config configs/config.yaml --out_dir outputs
```

功能：

```text
1. 扫描 raw/*.tif。
2. 解析 filename -> axis1..axis4。
3. 读取每张图的 shape, dtype, min, max, p0.5, p1, p50, p99, p99.5。
4. 输出 outputs/inspect/metadata.csv。
5. 输出 outputs/inspect/previews/*.png。
6. 输出 outputs/inspect/contact_sheet.png。
```

---

### 7.2 `scripts/01_register_groups.py`

命令：

```bash
python scripts/01_register_groups.py --raw_dir raw --config configs/config.yaml --out_dir outputs
```

功能：

```text
1. 读取 config。
2. 只处理 enabled=true 的 groups。
3. 对每组图片归一化。
4. 配准到 0000.tif。
5. 输出 aligned npy/png。
6. 输出 transforms.csv 和 QC 图。
```

---

### 7.3 `scripts/02_fuse_delined.py`

命令：

```bash
python scripts/02_fuse_delined.py --config configs/config.yaml --out_dir outputs
```

功能：

```text
1. 加载 enabled groups 的 aligned npy。
2. 每组计算 median/mean/std/mad。
3. 输出每组 fused image 和 line risk。
4. 合并 axis2_x 和 axis4_y 得到 combined_fused_delined。
5. 合并 line risk 得到 combined_line_risk。
```

---

### 7.4 `scripts/03_detect_candidates.py`

命令：

```bash
python scripts/03_detect_candidates.py --config configs/config.yaml --out_dir outputs
```

功能：

```text
1. 读取 combined_fused_delined.npy。
2. 读取 combined_line_risk.npy。
3. 读取 aligned images。
4. 检测 bright/dark 3–10 px blob。
5. 计算每个候选的局部对比度、圆度、长宽比、persistence、line_risk、combined_score。
6. 输出 outputs/candidates/candidates.csv。
7. 输出 outputs/candidates/candidates_overlay.png。
```

---

### 7.5 `scripts/04_export_candidate_crops.py`

命令：

```bash
python scripts/04_export_candidate_crops.py --config configs/config.yaml --out_dir outputs
```

功能：

```text
1. 读取 candidates.csv。
2. 按 combined_score 排序。
3. 最多导出 config crop_export.max_crops 个候选。
4. 每个候选输出一张多图 crop sheet。
5. 更新 candidates.csv 的 crop_path。
```

---

### 7.6 `scripts/05_make_report.py`

命令：

```bash
python scripts/05_make_report.py --config configs/config.yaml --out_dir outputs
```

功能：

```text
生成 outputs/reports/report.html。
```

---

### 7.7 `scripts/run_stage1.py`

命令：

```bash
python scripts/run_stage1.py --raw_dir raw --config configs/config.yaml --out_dir outputs
```

功能：

```text
按顺序调用：
00_inspect_dataset
01_register_groups
02_fuse_delined
03_detect_candidates
04_export_candidate_crops
05_make_report
```

要求：

```text
1. 每一步打印清楚日志。
2. 如果某一步失败，打印失败脚本和错误原因。
3. 每一步完成后打印关键输出路径。
```

---

## 8. 输出路径总览

Stage 1 完成后至少应有：

```text
outputs/inspect/metadata.csv
outputs/inspect/contact_sheet.png

outputs/normalized/*.npy
outputs/normalized/*.png

outputs/aligned/axis2_x/*.npy
outputs/aligned/axis2_x/*.png
outputs/aligned/axis4_y/*.npy
outputs/aligned/axis4_y/*.png

outputs/registration_qc/axis2_x_transforms.csv
outputs/registration_qc/axis2_x_aligned_contact_sheet.png
outputs/registration_qc/axis2_x_before_after_diff_contact.png
outputs/registration_qc/axis4_y_transforms.csv
outputs/registration_qc/axis4_y_aligned_contact_sheet.png
outputs/registration_qc/axis4_y_before_after_diff_contact.png

outputs/fusion/axis2_x_fused_median.npy
outputs/fusion/axis2_x_fused_median.png
outputs/fusion/axis4_y_fused_median.npy
outputs/fusion/axis4_y_fused_median.png
outputs/fusion/combined_fused_delined.npy
outputs/fusion/combined_fused_delined.png

outputs/line_risk/axis2_x_line_risk.npy
outputs/line_risk/axis2_x_line_risk.png
outputs/line_risk/axis4_y_line_risk.npy
outputs/line_risk/axis4_y_line_risk.png
outputs/line_risk/combined_line_risk.npy
outputs/line_risk/combined_line_risk.png

outputs/candidates/candidates.csv
outputs/candidates/candidates_overlay.png

outputs/candidate_crops/*.png

outputs/reports/report.html
```

---

## 9. 可视化要求

所有 PNG 预览要注意：

```text
1. 原始计算保留 float32，不要因为保存 PNG 丢失数据。
2. PNG 只是给用户看。
3. preview 拉伸必须使用 robust percentile，避免被极端亮点影响。
4. overlay 上候选点用圆圈标出，并尽量写 candidate_id。
5. line_risk 图可以用灰度或 heatmap，但不能影响 npy 数据。
```

Matplotlib 要使用非交互后端：

```python
import matplotlib
matplotlib.use("Agg")
```

---

## 10. 算法注意事项

### 10.1 不要让 Kossel line 主导配准

配准时不要直接使用高对比度细节图。请使用：

```text
原图 -> percentile normalization -> Gaussian blur sigma=5 -> phase correlation / ECC
```

这样可以降低 Kossel line、小缺陷、噪声对配准的干扰。

### 10.2 不要简单删掉 high line-risk 区域

真实 defect 可能被 Kossel line 覆盖。line-risk 只能作为扣分项，不能作为硬删除条件。

### 10.3 不要把传统 line mask 当真值

Stage 1 不使用传统 line mask 作为训练标签。当前重点是多帧证据。

### 10.4 不要使用调过对比度的 PNG 作为输入

所有计算必须从 `raw/*.tif` 开始。

### 10.5 不要第一阶段做深度学习

第一阶段只做可解释 pipeline，目标是产生 pseudo clean image 和候选点证据。

---

## 11. 验收标准

Codex 完成后，用户应能运行：

```bash
pip install -r requirements.txt
python scripts/run_stage1.py --raw_dir raw --config configs/config.yaml --out_dir outputs
```

并得到：

```text
1. 程序不崩溃。
2. 文件名解析正确。
3. metadata.csv 中 11 张图都有 axis1..axis4。
4. axis2_x、axis4_y 两组配准完成。
5. transforms.csv 中 0000.tif 的 dx=0, dy=0。
6. aligned_contact_sheet 能看到图像已经对齐。
7. combined_fused_delined.png 能作为第一版去线预览。
8. combined_line_risk.png 能显示变化剧烈区域。
9. candidates_overlay.png 有候选点标记。
10. candidates.csv 至少包含所有要求字段。
11. candidate_crops 中每个候选有多图证据 sheet。
12. report.html 可以打开浏览。
```

如果候选点太多或太少，先不要大改代码，只调整 `configs/config.yaml` 中：

```text
blob_detection.threshold_rel
blob_detection.min_diameter_px
blob_detection.max_diameter_px
candidate_filter.max_candidates
candidate_filter.min_circularity
candidate_filter.max_aspect_ratio
```

---

## 12. README 要写清楚

请创建 `README.md`，包含：

```text
1. 项目目的。
2. raw 文件命名规则。
3. 安装依赖。
4. 一键运行命令。
5. 每一步脚本的独立运行命令。
6. 输出文件说明。
7. 如何判断结果好坏。
8. 第一阶段限制。
9. 下一阶段计划。
```

---

## 13. 用户第一轮重点查看哪些输出

完成后请提示用户优先查看：

```text
outputs/inspect/contact_sheet.png
outputs/registration_qc/axis2_x_aligned_contact_sheet.png
outputs/registration_qc/axis4_y_aligned_contact_sheet.png
outputs/fusion/combined_fused_delined.png
outputs/line_risk/combined_line_risk.png
outputs/candidates/candidates_overlay.png
outputs/candidate_crops/ 前 20 个 score 最高的 crop
outputs/reports/report.html
```

用户后续会把这些结果发给 ChatGPT 判断：

```text
1. 配准是否合格。
2. Kossel line 是否被多图融合减弱。
3. line_risk 是否合理。
4. candidate 是否更像 defect 还是线伪影。
5. 下一步是调参数、改配准、加 axis1，还是开始 Stage 2。
```

---

## 14. 开发建议

请按模块实现，不要把所有逻辑写在一个脚本里。

推荐顺序：

```text
1. filename_parser.py + 00_inspect_dataset.py
2. normalize.py
3. registration.py + 01_register_groups.py
4. fusion.py + line_risk.py + 02_fuse_delined.py
5. blob_detection.py + candidate_scoring.py + 03_detect_candidates.py
6. crop_export.py + 04_export_candidate_crops.py
7. report.py + 05_make_report.py
8. run_stage1.py
9. README.md
```

每完成一个脚本就运行一次，保证每一步输出存在，不要等全部写完才测试。

---

## 15. 最终交付内容

请最终交付：

```text
1. 完整 src/ 模块
2. 完整 scripts/ 脚本
3. configs/config.yaml
4. requirements.txt
5. README.md
6. 能成功运行的一键命令
7. 输出目录中的所有关键结果
```

---

## 16. 后续 Stage 2 预留方向

Stage 1 完成后，后续可能做：

```text
1. 启用 axis1_angle，用 similarity/rotation 配准增加证据。
2. 启用 axis3_mag，用 scale-aware 配准验证候选尺寸。
3. 基于 combined_fused_delined 生成 pseudo-clean target。
4. 基于人工复核的 candidate crop 形成 defect / artifact 分类训练集。
5. 训练单图 Kossel line removal 模型。
6. 训练 defect detector。
```

Stage 1 请不要提前实现这些，只要接口和目录结构方便后续扩展即可。
