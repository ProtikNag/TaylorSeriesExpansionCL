# Task: Modernize the visualization pipeline and add a results dashboard skill

## Context

I am a CS PhD student working on continual learning and second-order optimization. My figure pipeline currently lives in the skill `algorithmic-art-academic` and is executed by my `visualization` subagent. Figures are publication targets (NeurIPS, ICML, journals) and must remain rigorous. I also want every batch of results collected into a modern, self-contained HTML results page.

Read these files before changing anything, then confirm the paths you found:

- `~/.claude/skills/algorithmic-art-academic/SKILL.md` (global)
- `~/.claude/agents/visualization.md` or whichever file defines the visualization subagent (global)
- `.claude/skills/` and `.claude/agents/` in the current repository (project copies)

## Deliverables

1. Update `algorithmic-art-academic/SKILL.md` (global and project copy) per Section A.
2. Create a new skill `visualization-dashboard/SKILL.md` (global and project copy) per Section B.
3. Update the visualization subagent definition so it invokes both skills and follows the workflow in Section C.
4. Add a verification script for the dashboard per Section D.
5. Print a diff summary of every file created or modified.

Do not modify any other skill or agent.

---

## Section A. Figure skill update (PNG and SVG)

### A.1 Rendering stack

- Primary renderer is Plotly (`plotly.py`) with Kaleido for static export. Each figure is defined once as a Plotly figure object and exported to both `figure.png` (scale chosen so the output is equivalent to 300 dpi at the intended print width) and `figure.svg`.
- Fallback renderer is matplotlib, used only when Kaleido cannot produce a faithful export (dense LaTeX, ridge plots, custom geometry). When matplotlib is used, apply the existing rcParams block from the skill.
- Install and pin the required Python packages in the skill's setup section: `plotly`, `kaleido`, `imageio`, `numpy`, `pandas`. Prefer `pip install --user` or a project virtualenv. Do not require Node or npm for figure generation.
- Verify each exported PNG and SVG opens and is non-empty before reporting success.

### A.2 Visual standard (unchanged constraints)

- White background, left and bottom axes only, horizontal grid lines only, faint grid `#E9ECEF`.
- Palette in order `#2563EB #D97706 #059669 #DC2626 #7C3AED #0891B2 #BE185D #92400E`.
- Fonts Inter for labels, Source Serif 4 for titles, JetBrains Mono for numeric annotations.
- No rainbow or jet colormaps, no dark backgrounds, no Anthropic brand colors.

### A.3 Visual standard (new requirements, to make figures less plain)

- Direct labeling of series at line ends instead of a legend box whenever there are five or fewer series.
- Annotate the single most important result on each figure (best value, crossover point, convergence step) with a short callout.
- Uncertainty always shown when available (shaded band for mean plus or minus std, or error bars), never omitted.
- Consistent tick density, human readable tick formatting, and axis titles with units.
- Small multiples preferred over many overlapping lines.
- No decorative effects on paper figures (no gradients, glow, shadows, rounded plot frames). Modern means clean hierarchy, not ornamentation.

### A.4 Animation

When the data has a temporal or iterative axis (training curves over epochs, loss landscape evolving over consolidation steps, parameter trajectories), additionally produce an animated version using Plotly frames, exported as `figure.gif` (via imageio) and `figure.html` (interactive Plotly export with `include_plotlyjs="cdn"`). Static PNG and SVG show the final frame or a small multiple of selected frames. Only animate when the animation reveals something the static figure cannot.

### A.5 Manifest contract

Every figure generation appends one entry to `report/manifest.json`. Create the file if absent. Schema:

```json
{
  "id": "loss_curves_seed_sweep",
  "group": "training_dynamics",
  "group_title": "Training dynamics",
  "title": "Validation loss across seeds",
  "caption": "One sentence, at most 160 characters.",
  "details": "Full description. Method, dataset, model, seeds, hyperparameters, what to look for. Markdown allowed.",
  "seed": 0,
  "dataset": "Atari Breakout",
  "model": "DQN",
  "variants": [
    {"label": "Line", "png": "figures/loss_curves_seed_sweep_line.png", "svg": "figures/loss_curves_seed_sweep_line.svg", "html": null},
    {"label": "Small multiples", "png": "figures/loss_curves_seed_sweep_grid.png", "svg": "figures/loss_curves_seed_sweep_grid.svg", "html": null}
  ],
  "animation": {"gif": "figures/loss_curves_seed_sweep.gif", "html": "figures/loss_curves_seed_sweep.html"},
  "metrics": [{"name": "final val loss", "value": 0.412, "unit": ""}],
  "created": "2026-09-09T14:00:00"
}
```

Rules for `variants`. Add a second or third variant only when a different chart type gives a different perspective on the same data (distribution versus trend, absolute versus relative, bar versus pie for composition). Do not produce redundant variants. The first variant is the paper figure.

All paths are relative to `report/`. Figures are written under `report/figures/`.

---

## Section B. New skill `visualization-dashboard`

### B.1 Trigger

Use this skill whenever I ask for a dashboard, a results page, an HTML report, or after any figure batch is generated by the visualization agent. Output is always `report/index.html`, regenerated idempotently from `report/manifest.json`. Never hand-edit the HTML. Never require a build step.

### B.2 Stack

- Single self-contained HTML file. Assets referenced by relative path only.
- Tailwind CSS via the Play CDN, DaisyUI via CDN with a custom dark theme, Lucide icons via the CDN `lucide` script with `lucide.createIcons()`.
- Inter from Google Fonts, JetBrains Mono for numbers.
- Plotly.js from CDN for interactive embeds of any variant that has an `html` or `animation.html` entry (embed via iframe to keep the page self-contained and safe).
- Vanilla JavaScript only. No React, no shadcn, no npm.

### B.3 Layout and behavior

- Minimalist dark theme. Background near `#0B0F14`, surface `#111827`, border `#1F2937`, text `#E5E7EB`, muted `#9CA3AF`, accent the palette blue `#2563EB`. Generous whitespace, 8 px spacing scale, max content width 1280 px.
- Sticky left sidebar (collapses to a top bar under 1024 px) listing each `group_title` as an anchor link, with a Lucide icon per group and the active section highlighted on scroll.
- Header with project name, run date, and a compact stats row (number of figures, groups, seeds) using DaisyUI `stat`.
- One section per group. Inside, a responsive bento grid of DaisyUI `card` components. Each card contains the figure image (PNG for raster, SVG when present, white figure on a light card inset so the paper figure is displayed as is), the short caption, and a metrics chip row rendered in JetBrains Mono.
- If a figure has more than one variant, render a segmented control (DaisyUI `join` of buttons) above the image that switches the displayed variant with a 200 ms crossfade. Keyboard accessible.
- If an animation exists, add a play button that swaps the static image for the GIF, and an "Interactive" button that opens the Plotly HTML in a modal (DaisyUI `modal` with an iframe).
- An info button (Lucide `info`) on every card. On hover and on click it opens a popover (DaisyUI `dropdown` or `popover`) showing the full `details` markdown, seed, dataset, model, and file links. Hover shows it, click pins it.
- Download buttons for PNG and SVG on each card.
- Scroll triggered fade-and-rise entrance for cards using IntersectionObserver, 300 ms ease-out, disabled when `prefers-reduced-motion` is set.
- A search box in the sidebar that filters cards by title, caption, or group.
- Footer with generation timestamp and manifest path.

### B.4 Generator

Implement `report/build_dashboard.py` (pure Python, standard library plus `markdown` if available, with a plain text fallback). It reads `manifest.json`, groups by `group`, orders groups by first appearance, and writes `index.html` from a template embedded in the script. Type hints and docstrings on every function.

---

## Section C. Visualization subagent workflow

Update the subagent instructions to this order, without skipping steps.

1. Read `algorithmic-art-academic/SKILL.md` and `visualization-dashboard/SKILL.md`.
2. Generate figures per Section A, appending to the manifest.
3. Run `python report/build_dashboard.py`.
4. Run the verification in Section D. If it fails, fix and rerun before reporting.
5. Report the list of figures, their groups, and the dashboard path. Do not describe the figures at length.

---

## Section D. Verification

Create `report/verify_dashboard.py` using Playwright (install `playwright` and the Chromium browser if absent). It must check, and fail loudly on any failure:

- `index.html` loads with zero console errors and zero failed network requests for local assets.
- Every image path in the manifest resolves to an existing non-empty file, and every `<img>` on the page has a natural width greater than zero.
- Every variant switch button changes the displayed image.
- Every info popover opens.
- The page renders without horizontal overflow at 390 px, 768 px, and 1440 px widths.
- Save a screenshot of the page at 1440 px to `report/verification/dashboard.png` for my review.

Also validate `manifest.json` against the schema in A.5 before building.

---

## Constraints

- Do not change the palette, fonts, or white background of paper figures.
- Do not fetch or copy code from design showcase sites. Implement the look from the description above.
- Do not add motivational commentary in outputs. Keep agent reports terse.
- Do not introduce a Node build step anywhere.
- Ask me before installing anything outside `pip` and the Playwright browser download.

## Acceptance checklist

- [ ] Both skill files updated in global and project locations, with identical content.
- [ ] Visualization agent references both skills and the Section C workflow.
- [ ] A sample run on synthetic data produces at least one grouped figure with two variants and one animated figure, a valid manifest, `report/index.html`, and a passing verification run with the screenshot saved.
- [ ] Diff summary printed.
