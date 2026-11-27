#!/usr/bin/env python3
"""Build the self-contained HTML results dashboard from ``report/manifest.json``.

The generator is idempotent: given the same manifest it emits byte-identical
HTML apart from the generation timestamp in the footer. ``index.html`` is never
hand-edited; every visual change belongs in the template below.

Stack constraints (see ``visualization-dashboard/SKILL.md``):
  * single HTML file, assets referenced by relative path only
  * Tailwind Play CDN + DaisyUI CDN + Lucide CDN, custom dark theme
  * vanilla JavaScript only, no build step, no npm

Usage::

    python report/build_dashboard.py [--manifest report/manifest.json]
                                     [--out report/index.html]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:  # optional dependency, plain-text fallback below
    import markdown as _markdown
except ImportError:  # pragma: no cover - exercised only when markdown is absent
    _markdown = None


# ── Theme tokens ────────────────────────────────────────────────────────────
# Dashboard chrome is dark. The figures themselves stay on their white ground
# and are never inverted, tinted, or filtered.
THEME = {
    "bg": "#0B0F14",
    "surface": "#111827",
    "border": "#1F2937",
    "text": "#E5E7EB",
    "muted": "#9CA3AF",
    "accent": "#2563EB",
    "inset": "#FFFFFF",  # light inset the paper figure sits on
}

# DaisyUI v4 consumes its palette as bare OKLCH triplets (``L% C H``).
DAISY_VARS = {
    "--b1": "16.65% 0.012 254.17",  # base-100   #0B0F14
    "--b2": "21.01% 0.032 264.66",  # base-200   #111827
    "--b3": "27.81% 0.030 256.85",  # base-300   #1F2937
    "--bc": "92.76% 0.006 264.53",  # base-content #E5E7EB
    "--p": "54.61% 0.215 262.88",   # primary    #2563EB
    "--pc": "100% 0 0",             # primary-content
    "--n": "21.01% 0.032 264.66",   # neutral    #111827
    "--nc": "92.76% 0.006 264.53",
    "--a": "54.61% 0.215 262.88",
    "--ac": "100% 0 0",
    "--rounded-box": "0.75rem",
    "--rounded-btn": "0.5rem",
}

# Lucide icon per group, chosen by keyword with a deterministic fallback cycle.
# Names are restricted to long-standing lucide identifiers.
GROUP_ICON_RULES: list[tuple[tuple[str, ...], str]] = [
    (("train", "dynamic", "curve", "learning"), "activity"),
    (("forget", "retention", "stability", "plasticity"), "layers"),
    (("ablat", "hyperparam", "tuning"), "sliders-horizontal"),
    (("compar", "baseline", "versus", "vs"), "git-compare"),
    (("landscape", "loss", "geometry", "curvature"), "mountain"),
    (("matrix", "heatmap", "confusion"), "grid-3x3"),
    (("scal", "sweep", "budget"), "trending-up"),
    (("time", "runtime", "cost", "compute", "wall"), "timer"),
    (("distribution", "histogram", "spread", "variance"), "bar-chart-2"),
    (("policy", "agent", "reward", "return", "rl"), "target"),
    (("diagnos", "probe", "analysis", "gradient"), "microscope"),
]
FALLBACK_ICONS = ["bar-chart-2", "circle", "box", "hash", "zap"]


# ── Manifest loading and validation ─────────────────────────────────────────
def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the manifest and split it into top-level metadata and figure entries.

    Accepts either a bare JSON array of entries or an object of the form
    ``{"project": str, "run_date": str, "figures": [...]}``.

    Raises:
        FileNotFoundError: the manifest does not exist.
        ValueError: the manifest is not valid JSON or has an unusable shape.
    """
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if isinstance(payload, list):
        return {}, payload
    if isinstance(payload, dict) and isinstance(payload.get("figures"), list):
        meta = {k: v for k, v in payload.items() if k != "figures"}
        return meta, payload["figures"]
    raise ValueError(
        f"{path}: expected a JSON array of figures or an object with a "
        f"'figures' array, got {type(payload).__name__}"
    )


def validate_manifest(
    entries: list[dict[str, Any]], report_dir: Path, check_files: bool = True
) -> list[str]:
    """Validate figure entries against the manifest schema.

    Args:
        entries: figure entries as loaded from the manifest.
        report_dir: directory all manifest paths are relative to.
        check_files: when True, also require every referenced asset to exist
            and be non-empty.

    Returns:
        A list of human-readable error strings. Empty means the manifest is valid.
    """
    errors: list[str] = []
    seen_ids: set[str] = set()

    def require_str(entry_label: str, obj: dict[str, Any], key: str) -> None:
        value = obj.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{entry_label}: '{key}' must be a non-empty string")

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"figure[{index}]: expected an object")
            continue
        label = f"figure[{index}] id={entry.get('id', '<missing>')!r}"

        for key in ("id", "group", "group_title", "title", "caption", "details", "created"):
            require_str(label, entry, key)

        figure_id = entry.get("id")
        if isinstance(figure_id, str):
            if figure_id in seen_ids:
                errors.append(f"{label}: duplicate id")
            seen_ids.add(figure_id)

        caption = entry.get("caption")
        if isinstance(caption, str) and len(caption) > 160:
            errors.append(f"{label}: 'caption' is {len(caption)} chars, limit is 160")

        created = entry.get("created")
        if isinstance(created, str):
            try:
                datetime.fromisoformat(created)
            except ValueError:
                errors.append(f"{label}: 'created' is not an ISO 8601 timestamp")

        if entry.get("seed") is not None and not isinstance(entry["seed"], int):
            errors.append(f"{label}: 'seed' must be an int or null")
        for key in ("dataset", "model"):
            if entry.get(key) is not None and not isinstance(entry[key], str):
                errors.append(f"{label}: '{key}' must be a string or null")

        variants = entry.get("variants")
        if not isinstance(variants, list) or not variants:
            errors.append(f"{label}: 'variants' must be a non-empty list")
        else:
            for v_index, variant in enumerate(variants):
                v_label = f"{label} variant[{v_index}]"
                if not isinstance(variant, dict):
                    errors.append(f"{v_label}: expected an object")
                    continue
                require_str(v_label, variant, "label")
                if not isinstance(variant.get("png"), str) or not variant["png"]:
                    errors.append(f"{v_label}: 'png' must be a relative path string")
                if not isinstance(variant.get("svg"), str) or not variant["svg"]:
                    errors.append(f"{v_label}: 'svg' must be a relative path string")
                if variant.get("html") is not None and not isinstance(variant["html"], str):
                    errors.append(f"{v_label}: 'html' must be a string or null")

        animation = entry.get("animation")
        if animation is not None:
            if not isinstance(animation, dict):
                errors.append(f"{label}: 'animation' must be an object or null")
            elif not any(animation.get(k) for k in ("gif", "html")):
                errors.append(f"{label}: 'animation' needs at least a 'gif' or 'html' path")

        metrics = entry.get("metrics")
        if metrics is not None:
            if not isinstance(metrics, list):
                errors.append(f"{label}: 'metrics' must be a list or null")
            else:
                for m_index, metric in enumerate(metrics):
                    m_label = f"{label} metric[{m_index}]"
                    if not isinstance(metric, dict):
                        errors.append(f"{m_label}: expected an object")
                        continue
                    require_str(m_label, metric, "name")
                    if "value" not in metric:
                        errors.append(f"{m_label}: missing 'value'")

        if check_files:
            for rel in iter_asset_paths(entry):
                target = report_dir / rel
                if not target.exists():
                    errors.append(f"{label}: asset does not exist: {rel}")
                elif target.stat().st_size == 0:
                    errors.append(f"{label}: asset is empty: {rel}")

    return errors


def iter_asset_paths(entry: dict[str, Any]) -> Iterable[str]:
    """Yield every asset path referenced by a figure entry, relative to report/."""
    for variant in entry.get("variants") or []:
        if not isinstance(variant, dict):
            continue
        for key in ("png", "svg", "html"):
            value = variant.get(key)
            if isinstance(value, str) and value:
                yield value
    animation = entry.get("animation") or {}
    if isinstance(animation, dict):
        for key in ("gif", "html"):
            value = animation.get(key)
            if isinstance(value, str) and value:
                yield value


# ── Grouping and presentation helpers ───────────────────────────────────────
def group_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group figures by ``group``, ordered by first appearance in the manifest."""
    groups: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = entry["group"]
        if key not in groups:
            groups[key] = {
                "key": key,
                "title": entry.get("group_title") or key,
                "icon": icon_for_group(key, len(groups)),
                "figures": [],
            }
        groups[key]["figures"].append(entry)
    return list(groups.values())


def icon_for_group(group_key: str, position: int) -> str:
    """Pick a stable Lucide icon name for a group key."""
    lowered = group_key.lower()
    for keywords, icon in GROUP_ICON_RULES:
        if any(word in lowered for word in keywords):
            return icon
    return FALLBACK_ICONS[position % len(FALLBACK_ICONS)]


def render_details(text: str) -> str:
    """Render the ``details`` field to HTML, falling back to escaped paragraphs."""
    if _markdown is not None:
        return _markdown.markdown(text, extensions=["extra", "sane_lists"])
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    return "".join(
        f"<p>{html.escape(block).replace(chr(10), '<br>')}</p>" for block in blocks
    )


def format_metric_value(value: Any) -> str:
    """Format a metric value compactly for the monospace chip row."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.3g}"
        # %g flips to exponential once the exponent reaches the precision, which
        # turned a plain score of 15350.75 into "1.535e+04".
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4g}"
    return str(value)


def esc(value: Any) -> str:
    """HTML-escape a value for use in element text or a quoted attribute."""
    return html.escape("" if value is None else str(value), quote=True)


def json_attr(value: Any) -> str:
    """Serialize a value as JSON safe to embed inside a double-quoted attribute."""
    return html.escape(json.dumps(value, ensure_ascii=False), quote=True)


# ── HTML rendering ──────────────────────────────────────────────────────────
def render_metric_chip(metric: dict[str, Any]) -> str:
    """Render one metric as a monospace chip."""
    unit = metric.get("unit")
    unit_html = f"<i>{esc(unit)}</i>" if unit else ""
    return (
        '<span class="chip">'
        f'{esc(metric.get("name"))}'
        f'<b>{esc(format_metric_value(metric.get("value")))}</b>'
        f"{unit_html}"
        "</span>"
    )


def render_card(entry: dict[str, Any], index: int) -> str:
    """Render one figure card."""
    variants = entry["variants"]
    first = variants[0]
    display_src = first.get("svg") or first["png"]
    animation = entry.get("animation") or {}
    gif = animation.get("gif")
    interactive = animation.get("html") or first.get("html")

    search_blob = " ".join(
        str(part)
        for part in (entry["title"], entry["caption"], entry["group_title"], entry["group"])
    ).lower()

    metric_chips = "".join(render_metric_chip(m) for m in (entry.get("metrics") or []))
    metrics_html = (
        f'<div class="mt-3 flex flex-wrap gap-2">{metric_chips}</div>' if metric_chips else ""
    )

    if len(variants) > 1:
        buttons = "".join(
            f'<button type="button" class="join-item vbtn{" vbtn-active" if i == 0 else ""}" '
            f'role="tab" aria-selected="{"true" if i == 0 else "false"}" '
            f'tabindex="{0 if i == 0 else -1}" data-variant-index="{i}">{esc(v["label"])}</button>'
            for i, v in enumerate(variants)
        )
        variant_control = (
            '<div class="join mb-3" role="tablist" '
            f'aria-label="Chart variants for {esc(entry["title"])}">{buttons}</div>'
        )
    else:
        variant_control = ""

    media_buttons: list[str] = []
    if gif:
        media_buttons.append(
            f'<button type="button" class="mini-btn js-play" data-gif="{esc(gif)}" '
            'aria-pressed="false"><i data-lucide="play"></i><span>Play</span></button>'
        )
    if interactive:
        media_buttons.append(
            f'<button type="button" class="mini-btn js-interactive" data-src="{esc(interactive)}">'
            '<i data-lucide="maximize-2"></i><span>Interactive</span></button>'
        )
    media_bar = (
        f'<div class="flex flex-wrap gap-2">{"".join(media_buttons)}</div>' if media_buttons else ""
    )

    meta_rows = "".join(
        f'<div class="pop-row"><span>{esc(label)}</span><b>{esc(value)}</b></div>'
        for label, value in (
            ("Seed", entry.get("seed")),
            ("Dataset", entry.get("dataset")),
            ("Model", entry.get("model")),
            ("Created", entry.get("created")),
        )
        if value is not None
    )

    file_links = "".join(
        f'<a href="{esc(path)}" target="_blank" rel="noopener">{esc(Path(path).name)}</a>'
        for path in iter_asset_paths(entry)
    )

    return f"""
      <article class="fig-card card" data-figure-id="{esc(entry['id'])}"
               data-search="{esc(search_blob)}"
               data-variants="{json_attr(variants)}"
               style="--stagger:{index % 3}">
        <div class="card-body">
          <div class="flex items-start justify-between gap-3">
            <h3 class="fig-title">{esc(entry['title'])}</h3>
            <div class="pop-wrap">
              <button type="button" class="icon-btn js-info" aria-expanded="false"
                      aria-label="Details for {esc(entry['title'])}">
                <i data-lucide="info"></i>
              </button>
              <div class="popover" role="dialog"
                   aria-label="Details for {esc(entry['title'])}" hidden>
                <div class="pop-head">{esc(entry['title'])}</div>
                <div class="pop-md">{render_details(entry['details'])}</div>
                {f'<div class="pop-meta">{meta_rows}</div>' if meta_rows else ''}
                {f'<div class="pop-files">{file_links}</div>' if file_links else ''}
              </div>
            </div>
          </div>

          {variant_control}

          <button type="button" class="fig-inset js-zoom"
                  aria-label="Open {esc(entry['title'])} full size">
            <img class="fig-img" src="{esc(display_src)}"
                 alt="{esc(entry['title'])}" loading="lazy" decoding="async">
          </button>

          <p class="fig-caption">{esc(entry['caption'])}</p>
          {metrics_html}

          <div class="card-actions flex flex-wrap items-center gap-2">
            {media_bar}
            <div class="ml-auto flex gap-2">
              <a class="mini-btn js-dl-png" href="{esc(first['png'])}" download>
                <i data-lucide="download"></i><span>PNG</span></a>
              <a class="mini-btn js-dl-svg" href="{esc(first['svg'])}" download>
                <i data-lucide="download"></i><span>SVG</span></a>
            </div>
          </div>
        </div>
      </article>"""


def render_section(group: dict[str, Any]) -> str:
    """Render one group section containing a bento grid of figure cards."""
    cards = "".join(render_card(entry, i) for i, entry in enumerate(group["figures"]))
    count = len(group["figures"])
    return f"""
    <section id="group-{esc(group['key'])}" class="group-section scroll-mt-24 mb-16">
      <header class="mb-6 flex items-center gap-3">
        <span class="group-icon"><i data-lucide="{esc(group['icon'])}"></i></span>
        <h2 class="group-title">{esc(group['title'])}</h2>
        <span class="group-count">{count} figure{'s' if count != 1 else ''}</span>
      </header>
      <div class="bento">
        {cards}
      </div>
    </section>"""


def build_html(
    groups: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    meta: dict[str, Any],
    manifest_path: Path,
    generated_at: datetime,
) -> str:
    """Assemble the complete dashboard page."""
    project = meta.get("project") or Path.cwd().name
    run_date = meta.get("run_date") or max(
        (e.get("created", "") for e in entries), default=""
    )[:10]
    seeds = {e.get("seed") for e in entries if e.get("seed") is not None}

    nav_links = "".join(
        f'<a class="nav-link" href="#group-{esc(g["key"])}" data-group="{esc(g["key"])}">'
        f'<i data-lucide="{esc(g["icon"])}"></i>'
        f'<span class="truncate">{esc(g["title"])}</span>'
        f'<em>{len(g["figures"])}</em></a>'
        for g in groups
    )
    sections = "".join(render_section(g) for g in groups)
    daisy_vars = "".join(f"      {k}: {v};\n" for k, v in DAISY_VARS.items())

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(project)} · Results</title>

<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {{
    theme: {{
      extend: {{
        colors: {{
          ink: '{THEME["bg"]}', surface: '{THEME["surface"]}',
          hair: '{THEME["border"]}', body: '{THEME["text"]}',
          muted: '{THEME["muted"]}', accent: '{THEME["accent"]}'
        }},
        fontFamily: {{
          sans: ['Inter', 'system-ui', 'Helvetica', 'Arial', 'sans-serif'],
          mono: ['JetBrains Mono', 'Menlo', 'Consolas', 'monospace']
        }},
        maxWidth: {{ shell: '1280px' }}
      }}
    }}
  }};
</script>
<link href="https://cdn.jsdelivr.net/npm/daisyui@4.12.14/dist/full.min.css" rel="stylesheet" type="text/css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
  :root {{
    --dash-bg: {THEME["bg"]};
    --dash-surface: {THEME["surface"]};
    --dash-border: {THEME["border"]};
    --dash-text: {THEME["text"]};
    --dash-muted: {THEME["muted"]};
    --dash-accent: {THEME["accent"]};
    --dash-inset: {THEME["inset"]};
    --sp: 8px;
  }}
  [data-theme="dark"] {{
{daisy_vars}  }}

  *, *::before, *::after {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    background: var(--dash-bg);
    color: var(--dash-text);
    font-family: 'Inter', system-ui, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  img, svg, iframe {{ max-width: 100%; }}
  a {{ color: inherit; text-decoration: none; }}
  /* Lucide replaces each <i data-lucide> with <svg class="lucide lucide-NAME">,
     dropping the placeholder's own attributes. Every icon rule below therefore
     targets both the placeholder and the .lucide element that replaces it. */
  .lucide {{ flex: 0 0 auto; width: 16px; height: 16px; }}
  :focus-visible {{ outline: 2px solid var(--dash-accent); outline-offset: 2px; border-radius: 6px; }}

  /* ── Shell ─────────────────────────────────────────────────────────── */
  .shell {{ max-width: 1280px; margin: 0 auto; display: flex; align-items: flex-start; }}
  .sidebar {{
    position: sticky; top: 0; height: 100vh; width: 264px; flex: 0 0 264px;
    border-right: 1px solid var(--dash-border);
    padding: 24px 16px; overflow-y: auto; overflow-x: hidden;
  }}
  .content {{ flex: 1 1 auto; min-width: 0; padding: 32px 24px 64px; }}
  @media (max-width: 1023px) {{
    .sidebar {{ display: none; }}
    .content {{ padding: 24px 16px 48px; }}
  }}

  .brand {{ font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }}
  .brand small {{ display: block; font-weight: 400; font-size: 12px; color: var(--dash-muted); margin-top: 2px; }}

  .search-wrap {{ position: relative; margin: 24px 0 16px; }}
  .search-wrap i, .search-wrap .lucide {{
    position: absolute; left: 10px; top: 50%; transform: translateY(-50%);
    width: 15px; height: 15px; color: var(--dash-muted); pointer-events: none;
  }}
  .search-input {{
    width: 100%; padding: 8px 10px 8px 32px; font-size: 13px;
    color: var(--dash-text); background: var(--dash-surface);
    border: 1px solid var(--dash-border); border-radius: 8px; outline: none;
  }}
  .search-input::placeholder {{ color: var(--dash-muted); }}
  .search-input:focus {{ border-color: var(--dash-accent); }}

  .nav-label {{
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--dash-muted); margin: 8px 0 8px 2px;
  }}
  .nav-link {{
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px; margin-bottom: 2px; border-radius: 8px;
    font-size: 13px; color: var(--dash-muted);
    border-left: 2px solid transparent;
    transition: background 120ms ease, color 120ms ease;
  }}
  .nav-link i, .nav-link .lucide {{ width: 15px; height: 15px; flex: 0 0 15px; }}
  .nav-link em {{ margin-left: auto; font-style: normal; font-family: 'JetBrains Mono', monospace; font-size: 11px; opacity: .6; }}
  .nav-link:hover {{ background: var(--dash-surface); color: var(--dash-text); }}
  .nav-link.active {{ background: var(--dash-surface); color: var(--dash-text); border-left-color: var(--dash-accent); }}

  /* ── Mobile top bar ────────────────────────────────────────────────── */
  .topbar {{
    display: none; position: sticky; top: 0; z-index: 40;
    background: rgba(11,15,20,.92); backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--dash-border); padding: 12px 16px;
  }}
  @media (max-width: 1023px) {{ .topbar {{ display: block; }} }}
  .chiprow {{ display: flex; gap: 8px; overflow-x: auto; padding-bottom: 2px; scrollbar-width: none; }}
  .chiprow::-webkit-scrollbar {{ display: none; }}
  .chip-nav {{
    flex: 0 0 auto; display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 10px; font-size: 12px; white-space: nowrap;
    color: var(--dash-muted); background: var(--dash-surface);
    border: 1px solid var(--dash-border); border-radius: 999px;
  }}
  .chip-nav i, .chip-nav .lucide {{ width: 13px; height: 13px; }}
  .chip-nav.active {{ color: var(--dash-text); border-color: var(--dash-accent); }}

  /* ── Header ────────────────────────────────────────────────────────── */
  .page-title {{ font-size: 26px; font-weight: 700; letter-spacing: -0.02em; margin: 0; }}
  .page-sub {{ color: var(--dash-muted); font-size: 13px; margin-top: 6px; }}
  .statbar {{
    margin-top: 24px; background: var(--dash-surface);
    border: 1px solid var(--dash-border); border-radius: 12px; width: 100%;
  }}
  .statbar .stat {{ border-color: var(--dash-border) !important; padding: 16px 20px; }}
  .statbar .stat-title {{ color: var(--dash-muted); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }}
  .statbar .stat-value {{ font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 500; color: var(--dash-text); }}

  /* ── Sections and cards ────────────────────────────────────────────── */
  .group-icon {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 9px;
    background: var(--dash-surface); border: 1px solid var(--dash-border);
    color: var(--dash-accent);
  }}
  .group-icon i, .group-icon .lucide {{ width: 16px; height: 16px; }}
  .group-title {{ font-size: 17px; font-weight: 600; margin: 0; letter-spacing: -0.01em; }}
  .group-count {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--dash-muted); }}

  /* Bento grid: fills the row whatever the card count, never narrower than the
     viewport allows, so the paper figure inside stays legible. */
  .bento {{
    display: grid;
    gap: 24px;
    grid-template-columns: repeat(auto-fit, minmax(min(420px, 100%), 1fr));
    align-items: stretch;
  }}
  .fig-card {{
    background: var(--dash-surface);
    border: 1px solid var(--dash-border);
    border-radius: 12px;
    overflow: visible;
    transition: border-color 160ms ease, transform 160ms ease;
  }}
  .fig-card:hover {{ border-color: #2b3648; }}
  .fig-card .card-body {{ padding: 16px; display: flex; flex-direction: column; height: 100%; }}
  .fig-card .card-actions {{ margin-top: auto; padding-top: 16px; }}
  .fig-card.span-2 {{ grid-column: span 2 / span 2; }}
  @media (max-width: 899px) {{ .fig-card.span-2 {{ grid-column: span 1 / span 1; }} }}

  .fig-title {{ font-size: 14px; font-weight: 600; margin: 0; line-height: 1.35; }}
  .fig-caption {{ margin: 12px 0 0; font-size: 12.5px; line-height: 1.55; color: var(--dash-muted); }}

  /* The paper figure keeps its white ground: never inverted, tinted, filtered. */
  .fig-inset {{
    margin-top: 4px; padding: 10px; border-radius: 9px; width: 100%;
    background: var(--dash-inset); border: 1px solid var(--dash-border);
    display: flex; align-items: center; justify-content: center; min-height: 120px;
    cursor: zoom-in; appearance: none; font: inherit; text-align: inherit;
  }}
  .fig-inset:hover {{ border-color: var(--dash-accent); }}
  .fig-img {{ display: block; width: 100%; height: auto; opacity: 1; transition: opacity 200ms ease-out; }}

  .chip {{
    display: inline-flex; align-items: baseline; gap: 5px;
    padding: 3px 8px; border-radius: 6px; font-size: 11px;
    background: rgba(37,99,235,.10); border: 1px solid rgba(37,99,235,.25);
    color: var(--dash-muted); font-family: 'JetBrains Mono', monospace;
  }}
  .chip b {{ color: var(--dash-text); font-weight: 500; }}
  .chip i {{ font-style: normal; opacity: .7; }}

  .join .vbtn {{
    font-size: 11.5px; padding: 5px 11px; color: var(--dash-muted);
    background: var(--dash-bg); border: 1px solid var(--dash-border);
    cursor: pointer; transition: color 120ms ease, background 120ms ease;
  }}
  .join .vbtn:hover {{ color: var(--dash-text); }}
  .join .vbtn-active {{ color: #fff; background: var(--dash-accent); border-color: var(--dash-accent); }}

  .mini-btn {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 10px; font-size: 11.5px; border-radius: 7px;
    color: var(--dash-muted); background: var(--dash-bg);
    border: 1px solid var(--dash-border); cursor: pointer;
    transition: color 120ms ease, border-color 120ms ease;
  }}
  .mini-btn:hover {{ color: var(--dash-text); border-color: var(--dash-accent); }}
  .mini-btn i, .mini-btn .lucide {{ width: 13px; height: 13px; }}
  .icon-btn {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 7px; flex: 0 0 26px;
    color: var(--dash-muted); background: transparent;
    border: 1px solid var(--dash-border); cursor: pointer;
  }}
  .icon-btn:hover, .icon-btn[aria-expanded="true"] {{ color: var(--dash-text); border-color: var(--dash-accent); }}
  .icon-btn i, .icon-btn .lucide {{ width: 14px; height: 14px; }}

  /* ── Info popover ──────────────────────────────────────────────────── */
  .pop-wrap {{ position: relative; }}
  .popover {{
    position: absolute; right: 0; top: 32px; z-index: 30;
    width: min(340px, calc(100vw - 48px));
    max-height: 420px; overflow-y: auto;
    padding: 14px; border-radius: 10px;
    background: #0F1724; border: 1px solid var(--dash-border);
    box-shadow: 0 16px 40px rgba(0,0,0,.55);
    font-size: 12.5px; line-height: 1.6; color: var(--dash-muted);
  }}
  .popover[hidden] {{ display: none; }}
  .pop-head {{ font-size: 12px; font-weight: 600; color: var(--dash-text); margin-bottom: 8px; }}
  .pop-md p {{ margin: 0 0 8px; }}
  .pop-md code {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--dash-text); }}
  .pop-md ul, .pop-md ol {{ margin: 0 0 8px 16px; padding: 0; }}
  .pop-meta {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--dash-border); }}
  .pop-row {{ display: flex; justify-content: space-between; gap: 12px; font-size: 11.5px; padding: 2px 0; }}
  .pop-row b {{ font-family: 'JetBrains Mono', monospace; font-weight: 400; color: var(--dash-text); }}
  .pop-files {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--dash-border); display: flex; flex-wrap: wrap; gap: 6px; }}
  .pop-files a {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    padding: 2px 7px; border-radius: 5px; color: var(--dash-muted);
    border: 1px solid var(--dash-border);
  }}
  .pop-files a:hover {{ color: var(--dash-text); border-color: var(--dash-accent); }}

  /* ── Modal ─────────────────────────────────────────────────────────── */
  #embed-modal .modal-box {{
    max-width: 1100px; width: calc(100vw - 32px); background: var(--dash-surface);
    border: 1px solid var(--dash-border); padding: 0; border-radius: 12px;
  }}
  #embed-modal header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; border-bottom: 1px solid var(--dash-border); font-size: 13px;
  }}
  #embed-frame {{ width: 100%; height: min(70vh, 640px); border: 0; background: #fff; display: block; }}

  /* ── Entrance animation ────────────────────────────────────────────── */
  .reveal {{ opacity: 0; transform: translateY(12px); }}
  .reveal.in {{
    opacity: 1; transform: none;
    transition: opacity 300ms ease-out, transform 300ms ease-out;
    transition-delay: calc(var(--stagger, 0) * 60ms);
  }}
  @media (prefers-reduced-motion: reduce) {{
    html {{ scroll-behavior: auto; }}
    .reveal, .reveal.in {{ opacity: 1; transform: none; transition: none; }}
    .fig-img {{ transition: none; }}
  }}

  .empty-state {{ display: none; padding: 48px 0; text-align: center; color: var(--dash-muted); font-size: 13px; }}
  .footer {{
    margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--dash-border);
    font-size: 11.5px; color: var(--dash-muted);
    display: flex; flex-wrap: wrap; gap: 8px 20px; justify-content: space-between;
  }}
  .footer code {{ font-family: 'JetBrains Mono', monospace; }}
</style>
</head>
<body>

<div class="topbar">
  <div class="flex items-center justify-between gap-3 mb-3">
    <div class="brand">{esc(project)}<small>Results dashboard</small></div>
  </div>
  <div class="search-wrap" style="margin:0 0 12px;">
    <i data-lucide="search"></i>
    <input class="search-input js-search" type="search" placeholder="Filter figures…"
           aria-label="Filter figures">
  </div>
  <nav class="chiprow" aria-label="Sections (compact)">
    {"".join(f'<a class="chip-nav" href="#group-{esc(g["key"])}" data-group="{esc(g["key"])}"><i data-lucide="{esc(g["icon"])}"></i>{esc(g["title"])}</a>' for g in groups)}
  </nav>
</div>

<div class="shell">
  <aside class="sidebar" aria-label="Sections">
    <div class="brand">{esc(project)}<small>Results dashboard</small></div>
    <div class="search-wrap">
      <i data-lucide="search"></i>
      <input class="search-input js-search" type="search" placeholder="Filter figures…"
             aria-label="Filter figures">
    </div>
    <div class="nav-label">Groups</div>
    <nav>{nav_links}</nav>
  </aside>

  <main class="content">
    <header>
      <h1 class="page-title">{esc(project)}</h1>
      <p class="page-sub">Run {esc(run_date or "—")} · generated from
        <code class="font-mono">{esc(manifest_path.as_posix())}</code></p>
      <div class="statbar stats stats-vertical sm:stats-horizontal">
        <div class="stat">
          <div class="stat-title">Figures</div>
          <div class="stat-value">{len(entries)}</div>
        </div>
        <div class="stat">
          <div class="stat-title">Groups</div>
          <div class="stat-value">{len(groups)}</div>
        </div>
        <div class="stat">
          <div class="stat-title">Seeds</div>
          <div class="stat-value">{len(seeds) if seeds else "—"}</div>
        </div>
      </div>
    </header>

    <div style="height:40px"></div>
    {sections}
    <div class="empty-state js-empty">No figures match that filter.</div>

    <footer class="footer">
      <span>Generated {esc(generated_at.strftime("%Y-%m-%d %H:%M:%S"))}</span>
      <span>Manifest <code>{esc(manifest_path.as_posix())}</code></span>
    </footer>
  </main>
</div>

<dialog id="embed-modal" class="modal">
  <div class="modal-box">
    <header>
      <span id="embed-title">Interactive figure</span>
      <button type="button" class="mini-btn" id="embed-close">
        <i data-lucide="x"></i><span>Close</span></button>
    </header>
    <iframe id="embed-frame" title="Interactive figure" src="about:blank"></iframe>
  </div>
  <form method="dialog" class="modal-backdrop"><button aria-label="Close">close</button></form>
</dialog>

<script src="https://cdn.jsdelivr.net/npm/lucide@0.454.0/dist/umd/lucide.min.js"></script>
<script>
(function () {{
  'use strict';

  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function drawIcons() {{
    if (window.lucide && typeof window.lucide.createIcons === 'function') {{
      try {{ window.lucide.createIcons(); }} catch (e) {{ /* icon set mismatch is non-fatal */ }}
    }}
  }}

  // ── Variant switcher ─────────────────────────────────────────────────
  function readVariants(card) {{
    try {{ return JSON.parse(card.getAttribute('data-variants')) || []; }}
    catch (e) {{ return []; }}
  }}

  function selectVariant(card, index) {{
    var variants = readVariants(card);
    if (!variants[index]) return;
    var variant = variants[index];
    var img = card.querySelector('.fig-img');
    var next = variant.svg || variant.png;
    var buttons = card.querySelectorAll('.vbtn');

    for (var i = 0; i < buttons.length; i++) {{
      var on = (i === index);
      buttons[i].classList.toggle('vbtn-active', on);
      buttons[i].setAttribute('aria-selected', on ? 'true' : 'false');
      buttons[i].tabIndex = on ? 0 : -1;
    }}
    card.setAttribute('data-active-variant', String(index));

    var png = card.querySelector('.js-dl-png');
    var svg = card.querySelector('.js-dl-svg');
    if (png && variant.png) png.setAttribute('href', variant.png);
    if (svg && variant.svg) svg.setAttribute('href', variant.svg);

    var play = card.querySelector('.js-play');
    if (play) {{
      play.setAttribute('aria-pressed', 'false');
      var span = play.querySelector('span');
      if (span) span.textContent = 'Play';
    }}

    if (reduceMotion) {{ img.setAttribute('src', next); return; }}
    img.style.opacity = '0';
    window.setTimeout(function () {{
      img.setAttribute('src', next);
      img.style.opacity = '1';
    }}, 200);
  }}

  document.querySelectorAll('.fig-card').forEach(function (card) {{
    var buttons = card.querySelectorAll('.vbtn');
    buttons.forEach(function (button, index) {{
      button.addEventListener('click', function () {{ selectVariant(card, index); }});
      button.addEventListener('keydown', function (event) {{
        var delta = event.key === 'ArrowRight' ? 1 : (event.key === 'ArrowLeft' ? -1 : 0);
        if (!delta) return;
        event.preventDefault();
        var next = (index + delta + buttons.length) % buttons.length;
        buttons[next].focus();
        selectVariant(card, next);
      }});
    }});
  }});

  // ── Animation: GIF play toggle ───────────────────────────────────────
  document.querySelectorAll('.js-play').forEach(function (button) {{
    button.addEventListener('click', function () {{
      var card = button.closest('.fig-card');
      var img = card.querySelector('.fig-img');
      var label = button.querySelector('span');
      var playing = button.getAttribute('aria-pressed') === 'true';
      if (playing) {{
        var index = parseInt(card.getAttribute('data-active-variant') || '0', 10);
        var variant = readVariants(card)[index] || readVariants(card)[0];
        img.setAttribute('src', variant.svg || variant.png);
        button.setAttribute('aria-pressed', 'false');
        if (label) label.textContent = 'Play';
      }} else {{
        img.setAttribute('src', button.getAttribute('data-gif'));
        button.setAttribute('aria-pressed', 'true');
        if (label) label.textContent = 'Stop';
      }}
    }});
  }});

  // ── Interactive embed modal ──────────────────────────────────────────
  var modal = document.getElementById('embed-modal');
  var frame = document.getElementById('embed-frame');
  var modalTitle = document.getElementById('embed-title');

  function openEmbed(card, src, suffix) {{
    var title = card.querySelector('.fig-title');
    modalTitle.textContent = (title ? title.textContent.trim() : 'Figure') + (suffix || '');
    frame.setAttribute('src', src);
    if (typeof modal.showModal === 'function') modal.showModal();
  }}

  document.querySelectorAll('.js-interactive').forEach(function (button) {{
    button.addEventListener('click', function () {{
      openEmbed(button.closest('.fig-card'), button.getAttribute('data-src'), '');
    }});
  }});

  // Clicking the figure opens it full size in the same modal.
  document.querySelectorAll('.js-zoom').forEach(function (inset) {{
    inset.addEventListener('click', function () {{
      var img = inset.querySelector('.fig-img');
      if (img) openEmbed(inset.closest('.fig-card'), img.getAttribute('src'), ' · full size');
    }});
  }});
  document.getElementById('embed-close').addEventListener('click', function () {{ modal.close(); }});
  modal.addEventListener('close', function () {{ frame.setAttribute('src', 'about:blank'); }});

  // ── Info popover: hover opens, click pins ────────────────────────────
  var pinned = null;

  function closePopover(button) {{
    var pop = button.parentElement.querySelector('.popover');
    if (!pop || pinned === button) return;
    pop.hidden = true;
    button.setAttribute('aria-expanded', 'false');
  }}

  function openPopover(button) {{
    var pop = button.parentElement.querySelector('.popover');
    if (!pop) return;
    pop.hidden = false;
    button.setAttribute('aria-expanded', 'true');
  }}

  function unpin() {{
    if (!pinned) return;
    var button = pinned;
    pinned = null;
    closePopover(button);
  }}

  document.querySelectorAll('.js-info').forEach(function (button) {{
    var wrap = button.parentElement;
    wrap.addEventListener('mouseenter', function () {{ openPopover(button); }});
    wrap.addEventListener('mouseleave', function () {{ closePopover(button); }});
    button.addEventListener('focus', function () {{ openPopover(button); }});
    button.addEventListener('click', function (event) {{
      event.stopPropagation();
      if (pinned === button) {{ unpin(); return; }}
      unpin();
      pinned = button;
      openPopover(button);
    }});
  }});

  document.addEventListener('click', function (event) {{
    if (pinned && !pinned.parentElement.contains(event.target)) unpin();
  }});
  document.addEventListener('keydown', function (event) {{
    if (event.key === 'Escape') unpin();
  }});

  // ── Scroll-triggered entrance ────────────────────────────────────────
  var cards = Array.prototype.slice.call(document.querySelectorAll('.fig-card'));
  if (reduceMotion || !('IntersectionObserver' in window)) {{
    cards.forEach(function (card) {{ card.classList.add('reveal', 'in'); }});
  }} else {{
    cards.forEach(function (card) {{ card.classList.add('reveal'); }});
    var revealer = new IntersectionObserver(function (records) {{
      records.forEach(function (record) {{
        if (!record.isIntersecting) return;
        record.target.classList.add('in');
        revealer.unobserve(record.target);
      }});
    }}, {{ rootMargin: '0px 0px -8% 0px', threshold: 0.05 }});
    cards.forEach(function (card) {{ revealer.observe(card); }});
  }}

  // ── Active section highlighting ──────────────────────────────────────
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('[data-group]'));
  function setActive(key) {{
    navLinks.forEach(function (link) {{
      link.classList.toggle('active', link.getAttribute('data-group') === key);
    }});
  }}
  var sections = Array.prototype.slice.call(document.querySelectorAll('.group-section'));
  if ('IntersectionObserver' in window && sections.length) {{
    var visible = {{}};
    var spy = new IntersectionObserver(function (records) {{
      records.forEach(function (record) {{
        visible[record.target.id] = record.isIntersecting ? record.intersectionRatio : 0;
      }});
      var best = null, bestRatio = 0;
      sections.forEach(function (section) {{
        var ratio = visible[section.id] || 0;
        if (ratio > bestRatio) {{ bestRatio = ratio; best = section; }}
      }});
      if (best) setActive(best.id.replace(/^group-/, ''));
    }}, {{ rootMargin: '-80px 0px -55% 0px', threshold: [0, 0.15, 0.4, 0.75, 1] }});
    sections.forEach(function (section) {{ spy.observe(section); }});
    setActive(sections[0].id.replace(/^group-/, ''));
  }}

  // ── Search filter ────────────────────────────────────────────────────
  var emptyState = document.querySelector('.js-empty');
  function applyFilter(rawQuery) {{
    var query = (rawQuery || '').trim().toLowerCase();
    var shown = 0;
    sections.forEach(function (section) {{
      var visibleInSection = 0;
      section.querySelectorAll('.fig-card').forEach(function (card) {{
        var match = !query || (card.getAttribute('data-search') || '').indexOf(query) !== -1;
        card.style.display = match ? '' : 'none';
        if (match) {{ visibleInSection++; shown++; }}
      }});
      section.style.display = visibleInSection ? '' : 'none';
    }});
    if (emptyState) emptyState.style.display = shown ? 'none' : 'block';
  }}
  document.querySelectorAll('.js-search').forEach(function (input) {{
    input.addEventListener('input', function () {{
      applyFilter(input.value);
      document.querySelectorAll('.js-search').forEach(function (other) {{
        if (other !== input) other.value = input.value;
      }});
    }});
  }});

  drawIcons();
  window.addEventListener('load', drawIcons);
}})();
</script>
</body>
</html>
"""


# ── Entry point ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Validate the manifest and write the dashboard. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="report/manifest.json",
                        help="path to the figure manifest (default: report/manifest.json)")
    parser.add_argument("--out", default="report/index.html",
                        help="path to write the dashboard (default: report/index.html)")
    parser.add_argument("--skip-file-check", action="store_true",
                        help="validate schema only, do not require assets to exist")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    report_dir = manifest_path.parent

    try:
        meta, entries = load_manifest(manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print(f"error: {manifest_path} contains no figures", file=sys.stderr)
        return 1

    errors = validate_manifest(entries, report_dir, check_files=not args.skip_file_check)
    if errors:
        print(f"error: {manifest_path} failed validation ({len(errors)} problems):",
              file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        return 1

    groups = group_entries(entries)
    document = build_html(groups, entries, meta, manifest_path, datetime.now())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")

    print(f"dashboard: {len(entries)} figures in {len(groups)} groups -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
