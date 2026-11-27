"""Place a figure into a temporary copy of the paper, compile it, and render the page.

Used to check that a figure's text and marks are legible at the printed size before
the figure is accepted. The real paper directory is never modified.

Example:
    python figures/preview_in_paper.py figures/storyline/pdf/storyline.pdf --width full
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

INCLUDE_WIDTH = {"single": r"\columnwidth", "full": r"\textwidth"}


def inject_figure(intro_tex: Path, figure: Path, width: str) -> None:
    """Insert a labelled preview figure right after the introduction's section heading."""
    env = "figure*" if width == "full" else "figure"
    block = (
        f"\n\\begin{{{env}}}[t]\\centering\n"
        f"\\includegraphics[width={INCLUDE_WIDTH[width]}]{{{figure.resolve()}}}\n"
        f"\\caption{{PREVIEW: \\detokenize{{{figure.name}}} at {width} width.}}\\label{{fig:preview}}\n"
        f"\\end{{{env}}}\n"
    )
    text = intro_tex.read_text()
    # Place after the first \section{...}; fall back to the top of the file.
    match = re.search(r"\\section\*?\{[^}]*\}", text)
    cut = match.end() if match else 0
    intro_tex.write_text(text[:cut] + block + text[cut:])


def figure_page(aux: Path) -> int:
    """Read the page number of the preview figure from the LaTeX .aux file."""
    match = re.search(r"\\newlabel\{fig:preview\}\{\{[^}]*\}\{(\d+)\}", aux.read_text())
    if not match:
        raise RuntimeError("preview label not found in aux file; did the paper compile?")
    return int(match.group(1))


def main() -> None:
    """Build the preview PDF and a page PNG, and print their paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", type=Path, help="figure file (pdf preferred, png accepted)")
    parser.add_argument("--width", choices=INCLUDE_WIDTH, default="single")
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    parser.add_argument("--main", default="tnnls_main.tex")
    parser.add_argument("--out", type=Path, default=Path("report/verification/paper_preview"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="paper_preview_"))
    shutil.copytree(args.paper, work / "paper", ignore=shutil.ignore_patterns(".git"))
    inject_figure(work / "paper" / "sections" / "introduction.tex", args.figure, args.width)

    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", args.main],
        cwd=work / "paper", check=True, capture_output=True,
    )
    stem = Path(args.main).stem
    page = figure_page(work / "paper" / f"{stem}.aux")

    args.out.mkdir(parents=True, exist_ok=True)
    png_prefix = args.out / f"{args.figure.stem}_{args.width}_page{page}"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(args.dpi), "-f", str(page), "-l", str(page),
         "-singlefile", str(work / "paper" / f"{stem}.pdf"), str(png_prefix)],
        check=True,
    )
    shutil.copy(work / "paper" / f"{stem}.pdf", args.out / f"{args.figure.stem}_{args.width}_paper.pdf")
    print(f"{png_prefix}.png")
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
