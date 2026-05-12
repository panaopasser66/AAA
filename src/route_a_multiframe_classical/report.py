from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
from jinja2 import Template


_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 1.3 / Stage 2 Report</title>
<style>
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; color: #222; }
h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { margin-top: 18px; }
img.preview { max-width: 720px; border: 1px solid #ccc; display: block; margin: 4px 0; }
img.preview.lg { max-width: 960px; }
img.crop { width: 220px; border: 1px solid #ddd; margin: 4px; }
table { border-collapse: collapse; font-size: 12px; }
th, td { border: 1px solid #ccc; padding: 2px 6px; }
th { background: #f5f5f5; }
.section { margin-bottom: 28px; }
.candidate-grid { display: flex; flex-wrap: wrap; }
.candidate { margin: 4px; padding: 4px; border: 1px solid #eee; width: 240px; font-size: 12px; }
.cat-hc-clean { border-left: 4px solid #2c7; }
.cat-line-adj { border-left: 4px solid #f80; }
.cat-tex      { border-left: 4px solid #08c; }
.cat-rej-line { border-left: 4px solid #c33; }
.cat-rej-tiny { border-left: 4px solid #999; }
.too_aggressive { background: #ffd9d9; }
.side-by-side { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
.side-by-side > div { flex: 1 1 480px; min-width: 320px; }
code { background: #f3f3f3; padding: 1px 4px; }
.note { font-size: 12px; color: #666; }
</style>
</head>
<body>
<h1>Stage 1.3 / Stage 2 Report</h1>
<p>Output dir: <code>{{ out_dir }}</code></p>

<div class="section">
<h2>1. Mask coverage</h2>
<p class="note">Rows tagged <code>too_aggressive</code> exceed the configured threshold.</p>
{{ mask_coverage_html|safe }}
</div>

<div class="section">
<h2>2. Registration QC (incl. axis1_angle)</h2>
{% for g in enabled_groups %}
  <h3>{{ g.name }} ({{ g.transform_model }})</h3>
  <table class="kv"><tr><td>files</td><td>{{ g.files|join(', ') }}</td></tr></table>
  {{ g.transforms_html|safe }}
  {% if g.aligned_sheet %}<div><b>Aligned contact sheet</b><img class="preview lg" src="{{ g.aligned_sheet }}"></div>{% endif %}
  {% if g.diff_before %}<div><b>Difference before</b><img class="preview" src="{{ g.diff_before }}"></div>{% endif %}
  {% if g.diff_after %}<div><b>Difference after</b><img class="preview" src="{{ g.diff_after }}"></div>{% endif %}
  {% if g.aligned_std %}<div><b>Aligned std</b><img class="preview" src="{{ g.aligned_std }}"></div>{% endif %}
  {% if g.valid_region %}<div><b>Group valid region</b><img class="preview" src="{{ g.valid_region }}"></div>{% endif %}
  {% if g.dynamic_line_mask %}<div><b>Per-group dynamic line mask</b><img class="preview" src="{{ g.dynamic_line_mask }}"></div>{% endif %}
{% endfor %}
</div>

<div class="section">
<h2>3. Fused images: v1 vs v2</h2>
<div class="side-by-side">
{% for fi in fused_compare %}
  <div><b>{{ fi.title }}</b><img class="preview lg" src="{{ fi.path }}"></div>
{% endfor %}
</div>
{% for fi in per_group_fused %}
  <div><b>{{ fi.title }}</b><img class="preview" src="{{ fi.path }}"></div>
{% endfor %}
</div>

<div class="section">
<h2>4. Dynamic line risk</h2>
{% for fi in line_risk_images %}
  <div><b>{{ fi.title }}</b><img class="preview" src="{{ fi.path }}"></div>
{% endfor %}
</div>

<div class="section">
<h2>5. Static line: raw vs refined</h2>
<div class="side-by-side">
{% for fi in static_compare %}
  <div><b>{{ fi.title }}</b><img class="preview" src="{{ fi.path }}"></div>
{% endfor %}
</div>
</div>

<div class="section">
<h2>6. Combined exclusion: v1 (coarse) vs refined</h2>
<div class="side-by-side">
{% for fi in exclusion_compare %}
  <div><b>{{ fi.title }}</b><img class="preview" src="{{ fi.path }}"></div>
{% endfor %}
</div>
</div>

<div class="section">
<h2>7. Safe search coverage</h2>
<div class="side-by-side">
{% for fi in safe_images %}
  <div><b>{{ fi.title }}</b><img class="preview" src="{{ fi.path }}"></div>
{% endfor %}
</div>
</div>

{% for v in version_blocks %}
<div class="section">
<h2>{{ v.index }}. {{ v.label }} &mdash; high-confidence_clean ({{ v.counts.high_confidence_clean }})</h2>
{% if v.overlays.high_confidence_clean %}<img class="preview lg" src="{{ v.overlays.high_confidence_clean }}">{% endif %}
<p>CSV: <code>{{ v.csvs.high_confidence_clean }}</code></p>
{{ v.tables.high_confidence_clean|safe }}
<div class="candidate-grid">
{% for c in v.crops.high_confidence_clean %}
  <div class="candidate cat-hc-clean">
    <div>id={{ c.candidate_id }} pol={{ c.polarity }} d={{ "%.1f"|format(c.diameter_px) }}px</div>
    <div>x={{ "%.0f"|format(c.x) }} y={{ "%.0f"|format(c.y) }}</div>
    <div>score={{ "%.3f"|format(c.final_score) }} aniso={{ "%.2f"|format(c.anisotropy) }}</div>
    <div>dRefStat={{ "%.1f"|format(c.distance_to_refined_static_line_skeleton) }} dRefComb={{ "%.1f"|format(c.distance_to_refined_combined_line_exclusion) }}</div>
    <div>area={{ "%.0f"|format(c.area_px) }} circ={{ "%.2f"|format(c.circularity) }} aspect={{ "%.2f"|format(c.aspect_ratio) }}</div>
    <div>uncov={{ c.uncovered_frame_count }}/{{ c.valid_frame_count }} spot={{ c.spot_present_when_uncovered }}</div>
    {% if c.crop_path %}<img class="crop" src="{{ c.crop_path }}">{% endif %}
  </div>
{% endfor %}
</div>

<h3>{{ v.label }} &mdash; line_adjacent_uncertain ({{ v.counts.line_adjacent_uncertain }})</h3>
{% if v.overlays.line_adjacent_uncertain %}<img class="preview" src="{{ v.overlays.line_adjacent_uncertain }}">{% endif %}
<div class="candidate-grid">
{% for c in v.crops.line_adjacent_uncertain %}
  <div class="candidate cat-line-adj">
    <div>id={{ c.candidate_id }} score={{ "%.3f"|format(c.final_score) }}</div>
    <div>dRefComb={{ "%.1f"|format(c.distance_to_refined_combined_line_exclusion) }} meanLR={{ "%.2f"|format(c.mean_line_risk_in_patch) }}</div>
    {% if c.crop_path %}<img class="crop" src="{{ c.crop_path }}">{% endif %}
  </div>
{% endfor %}
</div>

<h3>{{ v.label }} &mdash; texture_uncertain ({{ v.counts.texture_uncertain }})</h3>
{% if v.overlays.texture_uncertain %}<img class="preview" src="{{ v.overlays.texture_uncertain }}">{% endif %}

<h3>{{ v.label }} &mdash; rejected_line_like ({{ v.counts.rejected_line_like }})</h3>
{% if v.overlays.rejected_line_like %}<img class="preview" src="{{ v.overlays.rejected_line_like }}">{% endif %}

<h3>{{ v.label }} &mdash; rejected_tiny_response ({{ v.counts.rejected_tiny_response }})</h3>

<h3>{{ v.label }} &mdash; parameter sweep</h3>
<table>
<tr><th>preset</th><th>passing candidates</th><th>CSV</th><th>overlay</th></tr>
{% for s in v.sweeps %}
<tr {% if 'sanity' in s.name %}class="too_aggressive"{% endif %}>
<td>{{ s.name }}{% if 'sanity' in s.name %} <span class="note">(debug only, not HC)</span>{% endif %}</td>
<td>{{ s.count }}</td>
<td><code>{{ s.csv }}</code></td>
<td>{% if s.overlay %}<a href="{{ s.overlay }}">view</a>{% endif %}</td>
</tr>
{% endfor %}
</table>
</div>
{% endfor %}

<div class="section">
<h2>{{ stage3_section_index }}. Stage 3 &mdash; manual review &amp; line removal</h2>
<p class="note">These outputs are pseudo-labels and intermediate cleaned images. NOT final defects.</p>

{% if stage3.review_top_counts %}
<h3>3.1 Manual review ranking (source: {{ stage3.review_source }})</h3>
<table>
<tr><th>top N</th><th>CSV</th><th>overlay</th><th>review index</th></tr>
{% for n in stage3.review_top_counts %}
<tr>
  <td>{{ n }}</td>
  <td><code>{{ stage3.review_csvs[n] }}</code></td>
  <td>{% if stage3.review_overlays[n] %}<a href="{{ stage3.review_overlays[n] }}">view</a>{% endif %}</td>
  <td>{% if stage3.review_indices[n] %}<a href="{{ stage3.review_indices[n] }}">open</a>{% endif %}</td>
</tr>
{% endfor %}
</table>
<p>Manual labels template: <code>{{ stage3.review_labels_template }}</code></p>
{% if stage3.review_overlays[stage3.review_top_counts[-1]] %}
<img class="preview lg" src="{{ stage3.review_overlays[stage3.review_top_counts[-1]] }}">
{% endif %}
{% endif %}

{% if stage3.narrow_masks %}
<h3>3.2 Narrow line masks (Stage 3B)</h3>
{% if stage3.narrow_mask_coverage_html %}{{ stage3.narrow_mask_coverage_html|safe }}{% endif %}
<div class="side-by-side">
{% for m in stage3.narrow_masks %}
  <div><b>{{ m.title }}</b><img class="preview" src="{{ m.path }}"></div>
{% endfor %}
</div>
{% endif %}

{% if stage3.comparison_sheets %}
<h3>3.3 Inpainting comparison sheets</h3>
<div class="side-by-side">
{% for c in stage3.comparison_sheets %}
  <div><b>{{ c.title }}</b><img class="preview lg" src="{{ c.path }}"></div>
{% endfor %}
</div>
{% endif %}

{% if stage3.cleaned_images %}
<h3>3.4 Cleaned image variants (gallery)</h3>
<div class="side-by-side">
{% for c in stage3.cleaned_images %}
  <div><b>{{ c.title }}</b><img class="preview" src="{{ c.path }}"></div>
{% endfor %}
</div>
{% endif %}

{% if stage3.cleaned_summary_rows %}
<h3>3.5 Blob detection on cleaned variants</h3>
<p class="note">Selected variants per <code>cleaned_candidate_detection.selected_variants</code>. Detections here are NOT final defects.</p>
<table>
<tr><th>variant</th><th>status</th><th>total</th><th>passes_shape_filter</th><th>outside_invalid</th><th>overlay</th><th>CSV</th></tr>
{% for r in stage3.cleaned_summary_rows %}
<tr>
  <td>{{ r.variant }}</td>
  <td>{{ r.status }}</td>
  <td>{{ r.total }}</td>
  <td>{{ r.passes_shape_filter }}</td>
  <td>{{ r.outside_invalid }}</td>
  <td>{% if r.overlay %}<a href="{{ r.overlay }}">view</a>{% endif %}</td>
  <td><code>{{ r.csv }}</code></td>
</tr>
{% endfor %}
</table>
{% if stage3.cleaned_overlay_preview %}
<p class="note">Preview of one variant overlay:</p>
<img class="preview lg" src="{{ stage3.cleaned_overlay_preview }}">
{% endif %}
{% endif %}

<h3>3.6 Stage 3 caveats</h3>
<ol class="note">
<li>No deep learning is being used at this stage.</li>
<li>Stage 3A is manual review ranking.</li>
<li>Stage 3B is a traditional inpainting / line-removal baseline.</li>
<li>Candidates detected on cleaned images are NOT final defects; manual review is still required.</li>
<li>Manual labels collected here will feed Stage 4 deep-learning training.</li>
</ol>
</div>

<div class="section">
<h2>{{ dataset_section_index }}. Dataset overview</h2>
<p>Total TIFFs: {{ n_images }}</p>
{{ metadata_html|safe }}
{% if inspect_sheet %}<div><img class="preview lg" src="{{ inspect_sheet }}"></div>{% endif %}
</div>

</body>
</html>
"""


def _rel(p: str, start: str) -> str:
    if not p:
        return ""
    try:
        return os.path.relpath(p, start).replace("\\", "/")
    except ValueError:
        return p.replace("\\", "/")


def build_report(
    out_dir: str,
    metadata_csv: str,
    inspect_sheet: str,
    enabled_groups_meta: List[dict],
    mask_coverage_csv: str,
    fused_compare: List[dict],
    per_group_fused: List[dict],
    line_risk_images: List[dict],
    static_compare: List[dict],
    exclusion_compare: List[dict],
    safe_images: List[dict],
    version_blocks: List[dict],
    report_path: str,
    stage3: dict = None,
) -> str:
    report_dir = os.path.dirname(report_path)
    os.makedirs(report_dir, exist_ok=True)

    metadata_df = pd.read_csv(metadata_csv) if os.path.isfile(metadata_csv) else pd.DataFrame()
    metadata_html = metadata_df.to_html(index=False) if not metadata_df.empty else ""

    # Mask coverage table (with too_aggressive flag)
    mc_html = ""
    if os.path.isfile(mask_coverage_csv):
        mdf = pd.read_csv(mask_coverage_csv)
        mdf["coverage_percent"] = mdf["coverage_percent"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "n/a")
        def _row_class(flag):
            return ' class="too_aggressive"' if isinstance(flag, str) and "too_aggressive" in flag else ""
        rows = []
        rows.append("<table><tr><th>group</th><th>name</th><th>coverage</th><th>flag</th><th>path</th></tr>")
        for _, r in mdf.iterrows():
            rows.append(
                f"<tr{_row_class(r['flag'])}><td>{r['group']}</td><td>{r['name']}</td>"
                f"<td>{r['coverage_percent']}</td><td>{r['flag']}</td>"
                f"<td><code>{r['path']}</code></td></tr>"
            )
        rows.append("</table>")
        mc_html = "".join(rows)

    def relfi(items):
        return [
            {"title": i["title"], "path": _rel(i["path"], report_dir) if os.path.isfile(i["path"]) else ""}
            for i in items
        ]

    enabled_rendered = []
    for g in enabled_groups_meta:
        df = pd.read_csv(g["transforms_csv"]) if os.path.isfile(g["transforms_csv"]) else pd.DataFrame()
        cols = [c for c in [
            "filename", "dx", "dy", "rotation_deg", "scale",
            "phase_score", "ecc_score", "method", "shift_exceeds_max",
            "diff_before_mean", "diff_after_mean",
        ] if c in df.columns]
        enabled_rendered.append({
            "name": g["name"],
            "transform_model": g["transform_model"],
            "files": g["files"],
            "transforms_csv": _rel(g["transforms_csv"], report_dir),
            "transforms_html": df[cols].to_html(index=False, float_format="%.4f") if cols else "",
            "aligned_sheet": _rel(g.get("aligned_sheet", ""), report_dir),
            "diff_before": _rel(g.get("diff_before", ""), report_dir),
            "diff_after": _rel(g.get("diff_after", ""), report_dir),
            "aligned_std": _rel(g.get("aligned_std", ""), report_dir),
            "valid_region": _rel(g.get("valid_region", ""), report_dir),
            "dynamic_line_mask": _rel(g.get("dynamic_line_mask", ""), report_dir),
        })

    version_blocks_rel = []
    index_start = 8  # Mask, regQC, fused, dynLR, static, exclusion, safe = sections 1..7
    for i, v in enumerate(version_blocks):
        v2 = dict(v)
        v2["index"] = index_start + i
        v2["overlays"] = {
            k: _rel(p, report_dir) if p and os.path.isfile(p) else ""
            for k, p in v["overlays"].items()
        }
        v2["csvs"] = {k: _rel(p, report_dir) if p else "" for k, p in v["csvs"].items()}
        v2["crops"] = {}
        for cat, items in v["crops"].items():
            out = []
            for c in items:
                c2 = dict(c)
                c2["crop_path"] = _rel(c2.get("crop_path", ""), report_dir) if c2.get("crop_path") else ""
                out.append(c2)
            v2["crops"][cat] = out
        v2["sweeps"] = []
        for s in v.get("sweeps", []):
            v2["sweeps"].append({
                "name": s["name"],
                "count": s.get("count", 0),
                "csv": _rel(s.get("csv", ""), report_dir),
                "overlay": _rel(s.get("overlay", ""), report_dir) if s.get("overlay") and os.path.isfile(s.get("overlay", "")) else "",
            })
        version_blocks_rel.append(v2)

    stage3_section_index = index_start + len(version_blocks_rel)

    s3 = {
        "review_source": "", "review_top_counts": [],
        "review_csvs": {}, "review_overlays": {}, "review_indices": {},
        "review_labels_template": "",
        "narrow_masks": [], "narrow_mask_coverage_html": "",
        "cleaned_images": [], "comparison_sheets": [],
        "cleaned_summary_rows": [], "cleaned_overlay_preview": "",
    }
    if stage3:
        s3["review_source"] = stage3.get("review_source", "")
        s3["review_top_counts"] = list(stage3.get("review_top_counts", []))
        s3["review_csvs"] = {
            n: _rel(p, report_dir) if p else "" for n, p in stage3.get("review_csvs", {}).items()
        }
        s3["review_overlays"] = {
            n: _rel(p, report_dir) if p and os.path.isfile(p) else ""
            for n, p in stage3.get("review_overlays", {}).items()
        }
        s3["review_indices"] = {
            n: _rel(p, report_dir) if p and os.path.isfile(p) else ""
            for n, p in stage3.get("review_indices", {}).items()
        }
        s3["review_labels_template"] = _rel(stage3.get("review_labels_template", ""), report_dir)
        s3["narrow_masks"] = relfi(stage3.get("narrow_masks", []))
        s3["cleaned_images"] = relfi(stage3.get("cleaned_images", []))
        s3["comparison_sheets"] = relfi(stage3.get("comparison_sheets", []))

        # Render mask coverage CSV inline
        cov_csv = stage3.get("narrow_mask_coverage_csv", "")
        if cov_csv and os.path.isfile(cov_csv):
            cov_df = pd.read_csv(cov_csv)
            cov_df["coverage_percent"] = cov_df["coverage_percent"].map(
                lambda v: f"{v:.2f}%" if pd.notna(v) else "n/a"
            )
            rows = ["<table><tr><th>name</th><th>radius</th><th>coverage</th><th>flag</th></tr>"]
            for _, r in cov_df.iterrows():
                cls = ' class="too_aggressive"' if isinstance(r.get("flag"), str) and "too_aggressive" in r["flag"] else ""
                rows.append(
                    f"<tr{cls}><td>{r['name']}</td><td>{r['radius_px']}</td>"
                    f"<td>{r['coverage_percent']}</td><td>{r.get('flag','')}</td></tr>"
                )
            rows.append("</table>")
            s3["narrow_mask_coverage_html"] = "".join(rows)
        rows = []
        for r in stage3.get("cleaned_summary_rows", []):
            rows.append({
                "variant": r.get("variant", ""),
                "total": r.get("total", 0),
                "in_valid_region": r.get("in_valid_region", 0),
                "overlay": _rel(r.get("overlay", ""), report_dir) if r.get("overlay") and os.path.isfile(r.get("overlay", "")) else "",
                "csv": _rel(r.get("csv", ""), report_dir),
            })
        s3["cleaned_summary_rows"] = rows
        s3["cleaned_overlay_preview"] = (
            _rel(stage3.get("cleaned_overlay_preview", ""), report_dir)
            if stage3.get("cleaned_overlay_preview") and os.path.isfile(stage3.get("cleaned_overlay_preview", ""))
            else ""
        )

    html = Template(_TEMPLATE).render(
        out_dir=out_dir,
        n_images=len(metadata_df),
        metadata_html=metadata_html,
        inspect_sheet=_rel(inspect_sheet, report_dir) if os.path.isfile(inspect_sheet) else "",
        enabled_groups=enabled_rendered,
        mask_coverage_html=mc_html,
        fused_compare=relfi(fused_compare),
        per_group_fused=relfi(per_group_fused),
        line_risk_images=relfi(line_risk_images),
        static_compare=relfi(static_compare),
        exclusion_compare=relfi(exclusion_compare),
        safe_images=relfi(safe_images),
        version_blocks=version_blocks_rel,
        stage3=s3,
        stage3_section_index=stage3_section_index,
        dataset_section_index=stage3_section_index + 1,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path
