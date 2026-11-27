# Publication figure style

Portable spec for paper-destined figures. Hand this file to a session and it can
reproduce the look without seeing the originals.

Stack: **Plotly + Kaleido**, exported to PNG (300 dpi) and SVG. Nothing else is
required. matplotlib works if the same rules are applied.

---

## 1. Palette

```python
AC = {
    "blue": "#2563EB", "amber": "#D97706", "green": "#059669", "red": "#DC2626",
    "violet": "#7C3AED", "teal": "#0891B2", "rose": "#BE185D", "sienna": "#92400E",
    "bg": "#FFFFFF", "surface": "#F8F9FA", "border": "#DEE2E6",
    "axis": "#495057", "grid": "#E9ECEF",
    "text_primary": "#212529", "text_muted": "#6C757D", "text_faint": "#ADB5BD",
}
HIGHLIGHT_OURS = "#EFF6FF"   # row band for our method
HIGHLIGHT_RIVAL = "#FEF3C7"  # row band for the method to beat
```

Series in order: blue, amber, green, red, violet, teal, rose, sienna. Never
rainbow or jet. Never pure black or pure white as an explicit value.

**Fixed role assignment.** Keep these stable across every figure in a set, so a
colour means one thing throughout:

| Role | Colour |
|------|--------|
| Ours / primary method | blue |
| The baseline being compared against | amber |
| Ceiling, upper bound, reference model | green |
| Failure, below threshold, regression | red |
| Neutral reference (a specialist, a prior) | `text_faint` grey |

## 2. Typography

| Use | Font | Size at print (IEEE) |
|-----|------|------|
| Axis titles, labels, legends, panel tags | Helvetica (Arial fallback) | 8–9 pt |
| Tick labels, value labels, numerals | Helvetica (Arial fallback) | 7–8 pt |
| Paper figures carry no title (the LaTeX caption does) | — | — |

One sans-serif face throughout, as recommended by the IEEE graphics guidelines. No
monospace or decorative faces in print. Nothing below 7 pt at final printed size.
Figures are drawn at the printed width (3.5 in single column, 7.16 in full width),
so the sizes above are the sizes a reader sees. Every figure is checked by placing
it in a temporary copy of the paper and rendering the page
(`figures/preview_in_paper.py`) before it is accepted.

## 3. Non-negotiables

- White background. Tufte spine (left and bottom only). Horizontal grid only, in
  `grid`, 0.6 px.
- Axis titles in `text_muted` or `text_primary`; tick labels in `text_muted`.
- No gradients, glows, drop shadows, 3-D, or textured grounds.
- **No title inside the artwork.** The LaTeX caption carries it; a baked-in title
  duplicates it and is cropped at typesetting. (Dashboard figures are the
  exception and may keep one.)
- **Never invent uncertainty.** Single seed means no error bars and a stated
  "single seed". Never combine two standard deviations without their covariance.
- **Never pad precision.** If a run reports 3 decimals, print 3 next to someone
  else's 4. Padding implies precision you do not have.

## 4. Marks: pick by what the reader must judge

| The question | The mark |
|---|---|
| How large is each value, across many? | Bar from zero |
| How large is each value, across **3–6**? | **Lollipop**: thin stem from zero, dot at the value |
| How far past a per-row target? | **Dumbbell**: bar from target → achieved, dot at achieved |
| Did it clear one shared bar? | Bars or lollipops + a dashed rule across the panel |
| How do two states compare? | Slope chart |
| Matrix of ratios | Heatmap with a **clamped, pivoted** scale (§6) |

**Lollipop rule.** Below about six categories, thick bars are mostly ink. A thin
stem (3 px at 40% alpha) with a white-ringed dot encodes exactly the same thing
and leaves the panel quiet enough to carry a **dashed reference rule** at the
subject's value — and that rule is what turns a list of numbers into a
comparison. Keep zero on the axis here: stem length is a real magnitude.

**Dumbbell rule.** When every row has its *own* reference, do not draw bars from
zero: the stretch from 0 to the reference compares nothing, and it forces a
shared axis that flattens the differences you care about. Run the bar from the
reference to the achieved value so its **length and direction are the margin**,
put a white-ringed dot at the achieved end, and mark the reference with a short
vertical gate tick. Dropping the zero baseline is safe here precisely because
length now encodes a *difference*, not a magnitude — say so in the footnote.

**A reference is a rule, not a bar.** A ceiling, threshold or target drawn as a
fourth bar reads as a fourth competitor. Draw it as a dashed line across the
panel and put its value in the panel subtitle.

**Spread goes under the margin bar**, in a 60%-lightened tint of the row colour,
so only the overhang past the mean shows. That overhang is the informative half.

## 5. Layout

- **One shared legend at the top, then delete per-panel tick labels.** Rotated
  category labels repeated across panels are the single worst thing you can do to
  a small-multiple figure.
- **Small multiples get their own axis when scales differ by more than ~10×.**
  The usual advice is a shared axis; it is wrong when one panel runs to 20 and
  another to 4000, which flattens four panels to nothing. Own axis per panel,
  always including zero, and say so.
- **Numeric column beside the graph.** For per-row results, a right-hand block of
  numbers (value, reference, ratio) in a narrow second subplot. The graph
  carries shape, the column carries exact values. Add a faint rail across each row
  so the eye tracks from mark to number.
- **Band the two rows that matter**: ours in `HIGHLIGHT_OURS`, the rival in
  `HIGHLIGHT_RIVAL`, with matching text colour. Everything else stays neutral.
- Headroom: `max(bar_max * 1.20, reference * 1.04)`. Giving a reference rule the
  same headroom as a bar leaves a third of the panel empty.

## 6. Colour scales for ratio data

Clamp at the semantically meaningful ceiling and pivot at the meaningful
threshold. A scale stretched to accommodate one 360% outlier drags a genuinely
good 82% into alarming red.

```python
PIVOT = 0.65  # the "acceptable" line, not the midpoint of the data
SCALE = [
    [0.00, "#DC2626"], [0.22, "#E8736F"], [0.45, "#F4B3AE"],
    [PIVOT, "#F2F3F5"],
    [0.78, "#B9CDF6"], [0.89, "#7CA2F0"], [1.00, "#2563EB"],
]
# z = min(value, 1.0); the PRINTED number is always the true value.
```

Cell text: white below 0.26 or above 0.93, `text_primary` between. Bold the
diagonal. Colourbar ticks label the clamp explicitly: `≥100%`.

## 7. Tables as figures

Booktabs, drawn with shapes and annotations on hidden axes:

- Three rules only: top (1.3 px `axis`), under the header (1.0 px), bottom
  (1.3 px). **No vertical rules, no row shading** except the two banded rows.
- Section labels in 9 px uppercase `text_muted`.
- Values right-aligned with tabular figures; labels left-aligned.
- `±std` as a 6.5 px `text_faint` span appended to the value, not its own column.
- Advance a cursor by each row's own height so a separator costs ~0.34 of a row,
  not a whole empty one.
- Mark any cell that is **not the same quantity** as its column with `†` and
  explain it in a footnote. Exclude such cells from the "best value" bolding.

## 8. Footnotes inside the figure

A figure that needs a caveat to avoid misleading gets one, in 7.5–8.5 px
`text_muted` under the bottom rule. Put there: what the normaliser is, what the
error bars mean, why a cell is empty, any measurement asymmetry, seed count.

Plotly does not reflow text — **hand-wrap with `<br>`** and indent continuations
with `&nbsp;&nbsp;&nbsp;`. Budget ~108 characters per line at 8.5 px in a 648 px
figure. Anchor at `xref="paper", x=0` and back out of the left margin with
`xshift=-<left_margin>` so it starts at the figure edge, not the plot edge.

## 9. Export

```python
CSS_DPI, PRINT_DPI = 96, 300
EXPORT_SCALE = PRINT_DPI / CSS_DPI          # 3.125
W_FULL, W_ONE_HALF = 648, 480               # 6.75 in, 5.00 in at 96 dpi

fig.write_image(png, width=W, height=H, scale=EXPORT_SCALE)
fig.write_image(svg, width=W, height=H)
```

Always both formats, in `png/` and `svg/` beside each other.

**The SVG must be genuinely vector. Verify it:**

```bash
grep -c '<image' fig.svg   # must be 0
```

`go.Heatmap` **rasterises into an embedded bitmap** and will fail this. For a
small matrix, draw each cell as a `add_shape(type="rect")` on numeric axes
(`range=[0, n]`, ticks re-labelled at cell centres `i + 0.5`), and carry the
colourbar on an invisible marker trace:

```python
go.Scatter(x=[None], y=[None], mode="markers", showlegend=False,
           marker=dict(color=[0], colorscale=SCALE, cmin=0, cmax=1,
                       showscale=True, opacity=0, colorbar=dict(...)))
```

Plotly renders that colourbar as an SVG gradient.

## 10. Data hygiene

- Keep numbers in a `data.json` beside the script, never inline in plotting code,
  with a `provenance` block naming the source file and any caveat.
- Read shared constants out of the codebase rather than retyping them; parse with
  `ast.literal_eval` if importing would drag in heavy deps.
- Recompute headline statistics from the raw data instead of copying them out of
  an older README. Stale captions that contradict their own figure are the most
  common defect in a results folder.

## 11. Checklist

- [ ] No title inside the artwork
- [ ] Palette roles consistent across the whole figure set
- [ ] One sans face (Helvetica/Arial), nothing below 7 pt at print size; checked in the paper preview
- [ ] Mark chosen by §4, not by habit
- [ ] Per-row references drawn as rules or dumbbell anchors, never extra bars
- [ ] Legend once; no rotated per-panel tick labels
- [ ] Ratio scales clamped and pivoted; printed numbers un-clamped
- [ ] Non-comparable cells daggered and excluded from bolding
- [ ] Uncertainty shown where repeats exist, absent and stated where they do not
- [ ] Footnote covers normaliser, error-bar meaning, empty cells, seed count
- [ ] PNG at scale 3.125 **and** SVG, both non-empty
- [ ] `grep -c '<image' *.svg` returns 0
