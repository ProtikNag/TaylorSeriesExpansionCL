# Pseudocode index

High-level algorithmic pseudocode for the implemented HTCL algorithms. PDFs are rendered with
`render_pdf.py` (matplotlib PdfPages fallback; the cluster's LaTeX install has no format file).
Rebuild a PDF with: `.venv/bin/python pseudocode/render_pdf.py IN.md OUT.pdf`.

| # | Algorithm | Source | Realizes | Summary |
|---|-----------|--------|----------|---------|
| 01 | HTCL — the algorithm (group schedule) | `htcl/hierarchy.py` (`learn`, `_learn_group`, `_consolidate`, `_merge_search`, `_consolidation_data`, `_bn_data`, `_taylor`; `_guard_consolidation` = V2 ablation); `htcl/consolidation.py` (`mean_gradient`, `fisher_diag`, `hutchinson_diag`, `taylor_step`, `recalibrate_bn`, `average_states`) | `docs/ALGORITHM.md`; eq. `hier_update` (hierarchical Taylor consolidation); §C15 within-window order invariance; §5 BN/gradient buffer split | Per group of k tasks: k! order search picks the best-validation local, then the closed-form Taylor step $w_i \gets w_i + \eta_i(H_i+\lambda_i I)^{-1}(\lambda_i\Delta d_i - g_i)$ cascades it as a *direction* and a line search over $\alpha\in\{0,.25,.5,.75,1\}$ plus the local picks the merge on all-seen-task validation (`merge_search`); a short final group is merged the same way (`merge_last`); the Taylor gradient excludes HTCL's reservoir while BN recalibration includes it. V1 (`consolidate:false`) and V2 (`merge_search:false,guard:true`, all-or-nothing) are ablations. |
