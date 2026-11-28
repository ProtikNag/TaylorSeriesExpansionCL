# Project rules (HTCL, TNNLS revision)

Rules for any Claude session working in this repository. They were set by the first author during the
revision and bind every change, run, figure and text. Read `docs/HANDOFF.md` (current state),
`README.md` (how to run) and `docs/DESIGN.md` (the binding code contract) before acting.

## Context to load when it is present

These folders are tracked in this repository for now (the author's choice). They must be removed from the
anonymous release copy before submission, and `scripts/check_anonymity.sh` skips them.

- `paper/`: the TNNLS manuscript (`tnnls_main.tex`, `sections/*.tex`; compile with `latexmk -pdf tnnls_main.tex`).
- `reviews/`:
  - `neurips2026/reviews.md`: the rejected NeurIPS 2026 reviews, verbatim.
  - `CONCERNS.md`: the reviewer concerns C1 to C17.
  - `PLAN.md`: the revision plan, decisions and the current storyline.
  - `compute_options.md`: compute and cost plan.
- `example/`: the authors' TMLR submission (`14677_Forgetting_Returns_with_.pdf`) and its visualization
  guide (`visualization-dashboard-prompt copy.md`). The paper's organization and figures follow it. Its
  code and figure scripts live in the TMLR repository next to this one: `../CRL-Minmax` on the cluster,
  `~/Desktop/Continual Reinforcement Learning with Minimax Algorithm` on the laptop. In that repository,
  `reports/final/motivation/make_figures.py` drew its intro figure.

## Experiments

- **One training seed for the whole paper** (`PAPER_SEED` in `htcl/run.py`): the same for every method
  and task order. Never add seed sweeps or seed controls. The spread across task orders is the statistic.
- **Task orders:**
  - 20 per experiment in the full protocol (all t! when t! < 20).
  - The storyline uses 5.
  - When a figure needs more orders, grow in steps (3, 5, 7, 10), never straight to the maximum.
- **Equal deployed memory.** HTCL's deployed model and every baseline hold the same replay buffer N.
  - The k-task window HTCL's local learner trains on is training-time memory, reported as
    `htcl_window_tasks`.
  - For methods without a buffer (finetune, EWC, SR), HTCL supplies the N-sample buffer. The paper
    states this: without replay these methods cannot retain memory.
- **HTCL is the group schedule (k=2), the algorithm the paper describes:**
  - The stream is split into consecutive groups of k tasks. For each group a local learner is trained
    from the persistent hierarchical model w_1, trying both orders of the pair and keeping the better one
    on validation data; finetune/EWC/SR get HTCL's N-sample reservoir replayed during local training.
  - The global is then merged toward that local with the closed-form Taylor step, kept under a safety
    check (guard) or a step-size search (`merge_search`) over α ∈ {0, .25, .5, .75, 1}.
  - The online schedule (`schedule: online`, consolidate every task, revise every k) is a documented
    ablation only. Catch-up is not part of the algorithm.
  - λ defaults to 1/lr.
- **Selection** uses the 10% validation split. Test data is for reporting only.
- **Resumable runs.**
  - Every job checkpoints after each task (atomically, with RNG state) and writes `DONE` when finished.
  - Rerunning the same command resumes; never delete a run directory to "restart".
  - The job id hashes every resolved hyperparameter and the seed, so changed settings never reuse old
    results.
  - A diverged job fails loudly and does not write chance-level metrics.
- **Reduced-data runs** (`--samples-per-class`) validate code paths. They are never paper numbers.
- **Where things run.**
  - All training runs happen on the cluster; the laptop freezes under them.
  - The cluster session never makes figures or other visualizations, because its terminal cannot show
    them. It pushes numbers to the `cluster/results` branch (`docs/HANDOFF.md`, "Reporting back").
  - The laptop session pulls those numbers, draws the figures, and checks them in the paper.
  - Results compared in one figure or table come from the same device.
- **Before running anything large**, show the user the list of experiments and variations first.
- Never change a config's hyperparameters silently. Report the problem and propose the change.

## Figures

- Follow the TMLR kit:
  - `report/acviz.py` (Plotly) or `report/paperfig.py` (matplotlib)
  - export PNG at 300 dpi plus SVG and PDF
  - register the figure in `report/manifest.json`
  - `report/build_dashboard.py` and `report/verify_dashboard.py`
- Style is in `figures/FIGURE_STYLE.md`:
  - academic palette, white background, Tufte spine, no title inside the figure
  - Helvetica with Arial fallback, nothing below 7 pt at printed size
  - drawn at IEEE width (3.5 in column, 7.16 in text width)
- **Every new figure goes into the paper for a check:**
  `python figures/preview_in_paper.py FIG.pdf --width single|full`. Look at the rendered page and fix
  sizes and fonts before accepting it.
- **Independent agents review each figure** against a TNNLS/TMLR acceptance bar before it is shown as done.
- **The storyline figure (intro Fig. 1) stays simple.**
  - One column per method, one dot per task order, a worst-to-best line, and the gap Δ.
  - Families: no protection, replay, regularization.
  - The HTCL version puts each method beside its HTCL variant.
  - The user rejected order-marker shapes, schematic side panels and density plots.

## Writing (paper text and reports)

- Formal, direct academic prose. Use colons and em dashes sparingly. Never use "straightforward",
  "genuinely" or "honestly".
- Equations in LaTeX. Give time and space complexity when discussing algorithms.
- When reviewing drafts, flag unsupported claims.

## Code and repository

- Python with type hints and docstrings; PyTorch.
- Run `.venv/bin/python -m pytest -q tests` after code changes.
- Commit or push only when the user asks.
- Never commit data, `runs/`, checkpoints, model files or fonts.
- `paper/`, `reviews/` and `example/` are tracked for now. Before submission, remind the author to:
  - make the repository private;
  - create an anonymized code repository, as for the TMLR submission;
  - drop those folders from it;
  - run a thorough review.
- Run `scripts/check_anonymity.sh` before every commit. Repository text must not name people,
  institutions or clusters. The release will be an anonymized copy.
- Ask before installing anything that is not a pip package.
