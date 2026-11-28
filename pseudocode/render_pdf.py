#!/usr/bin/env python
"""Render a lightweight pseudocode markdown file to a paginated PDF via matplotlib.

Fallback renderer for hosts whose LaTeX install is broken (no format file). Supports a small
markdown subset tuned for algorithm pseudocode:

    # Title              -> page/section heading (large bold)
    ## Heading           -> sub-heading (bold)
    **bold** inline      -> bold run
    `code` inline        -> monospace run
    - bullet / 1. item   -> indented list line
    > note               -> indented note line
    ```...```            -> verbatim monospace block (kept as-is)
    plain text           -> body line; inline $...$ is passed to matplotlib mathtext

Indentation (leading spaces) in the source is preserved for pseudocode nesting.

Usage: python render_pdf.py IN.md OUT.pdf
"""
from __future__ import annotations

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Page geometry (inches) and typography.
PAGE_W, PAGE_H = 8.5, 11.0
MARGIN_L, MARGIN_R = 0.85, 0.6
MARGIN_T, MARGIN_B = 0.75, 0.75
LINE_H = 0.0175          # vertical step per line as a fraction of the axes height
FONT_BODY = 9.0
FONT_H1 = 15.0
FONT_H2 = 11.5
FONT_MONO = 8.3
CHAR_W = 0.0092         # approx width of one body char as a fraction of axes width (for wrapping)
WRAP_COLS = int((1.0) / CHAR_W)  # usable columns across the text area


def _emit_segments(line: str):
    """Split a line into (text, style) runs honoring **bold**, `code`, and $math$.

    Returns a list of (text, kind) where kind in {body, bold, mono, math}.
    """
    out = []
    # First protect $...$ math spans, then split the rest on ** and `.
    tokens = re.split(r"(\$[^$]*\$)", line)
    for tok in tokens:
        if tok.startswith("$") and tok.endswith("$") and len(tok) >= 2:
            out.append((tok, "math"))
            continue
        # bold / code within non-math text
        for piece in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", tok):
            if not piece:
                continue
            if piece.startswith("**") and piece.endswith("**"):
                out.append((piece[2:-2], "bold"))
            elif piece.startswith("`") and piece.endswith("`"):
                out.append((piece[1:-1], "mono"))
            else:
                out.append((piece, "body"))
    return out


def _wrap(text: str, width: int, indent: str):
    """Greedy word wrap preserving a leading indent on continuation lines."""
    words = text.split(" ")
    lines, cur = [], indent
    for w in words:
        trial = (cur + w) if cur.endswith(" ") or cur == indent else (cur + " " + w)
        if len(trial) > width and cur.strip():
            lines.append(cur)
            cur = indent + w
        else:
            cur = trial
    lines.append(cur)
    return lines


class Renderer:
    def __init__(self, path: str):
        self.pdf = PdfPages(path)
        self._new_page()

    def _new_page(self):
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.ax = self.fig.add_axes([
            MARGIN_L / PAGE_W, MARGIN_B / PAGE_H,
            1 - (MARGIN_L + MARGIN_R) / PAGE_W, 1 - (MARGIN_T + MARGIN_B) / PAGE_H,
        ])
        self.ax.axis("off")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.y = 1.0

    def _space(self, dh: float):
        self.y -= dh
        if self.y < 0.0:
            self._flush()
            self._new_page()

    def _flush(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def _draw_runs(self, runs, x0: float, size: float):
        """Draw a sequence of styled runs on the current line, left to right."""
        x = x0
        for text, kind in runs:
            if kind == "math":
                fam, weight, fp = "serif", "normal", text
            elif kind == "bold":
                fam, weight, fp = "sans-serif", "bold", text
            elif kind == "mono":
                fam, weight, fp = "monospace", "normal", text
            else:
                fam, weight, fp = "sans-serif", "normal", text
            try:
                self.ax.text(x, self.y, fp, fontsize=size, family=fam, fontweight=weight,
                             ha="left", va="top", transform=self.ax.transAxes)
            except Exception:
                # mathtext could not parse this run; draw it verbatim without math rendering.
                self.ax.text(x, self.y, fp.replace("$", ""), fontsize=size, family="monospace",
                             ha="left", va="top", transform=self.ax.transAxes)
            # Advance x by an estimate of the run width (chars * char width at this size).
            est = len(text) if kind != "math" else max(1, len(text) - 2)
            x += est * CHAR_W * (size / FONT_BODY)

    def line(self, raw: str):
        stripped = raw.rstrip("\n")
        indent_n = len(stripped) - len(stripped.lstrip(" "))
        indent = " " * indent_n
        content = stripped.strip()

        if content == "":
            self._space(LINE_H * 0.6)
            return
        if content.startswith("# "):
            self._space(LINE_H * 1.4)
            self.ax.text(0, self.y, content[2:], fontsize=FONT_H1, family="sans-serif",
                         fontweight="bold", ha="left", va="top", transform=self.ax.transAxes)
            self._space(LINE_H * 2.2)
            self.ax.plot([0, 1], [self.y + LINE_H, self.y + LINE_H], color="0.3", lw=0.8,
                         transform=self.ax.transAxes)
            self._space(LINE_H * 0.3)
            return
        if content.startswith("## "):
            self._space(LINE_H * 1.2)
            self.ax.text(0, self.y, content[3:], fontsize=FONT_H2, family="sans-serif",
                         fontweight="bold", ha="left", va="top", transform=self.ax.transAxes)
            self._space(LINE_H * 1.9)
            return

        x0 = indent_n * CHAR_W
        avail = max(20, WRAP_COLS - indent_n)
        # Wrap on the plain text length, then re-segment each wrapped physical line.
        plain = re.sub(r"\*\*|`|\$", "", content)
        wrapped = _wrap(plain, avail, "")
        # Only the first physical line carries the real markup; for simplicity re-segment the
        # whole content once and place it on the first wrapped line, continuation lines plain.
        if len(wrapped) == 1:
            self._draw_runs(_emit_segments(content), x0, FONT_BODY)
            self._space(LINE_H)
        else:
            # Multi-line: render wrapped plain text (markup stripped) to keep layout correct.
            for i, wl in enumerate(wrapped):
                self._draw_runs(_emit_segments(wl), x0 + (0 if i == 0 else 2 * CHAR_W), FONT_BODY)
                self._space(LINE_H)

    def verbatim(self, text: str):
        for ln in text.split("\n"):
            self.ax.text(0.01, self.y, ln, fontsize=FONT_MONO, family="monospace",
                         ha="left", va="top", transform=self.ax.transAxes)
            self._space(LINE_H)

    def close(self):
        self._flush()
        self.pdf.close()


def main(inp: str, outp: str):
    with open(inp, encoding="utf-8") as f:
        src = f.read()
    r = Renderer(outp)
    in_block, block = False, []
    for raw in src.split("\n"):
        if raw.strip().startswith("```"):
            if in_block:
                r.verbatim("\n".join(block))
                block, in_block = [], False
            else:
                in_block = True
            continue
        if in_block:
            block.append(raw)
        else:
            r.line(raw)
    r.close()
    print(f"wrote {outp}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: render_pdf.py IN.md OUT.pdf")
    main(sys.argv[1], sys.argv[2])
