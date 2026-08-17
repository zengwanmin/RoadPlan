# RoadPlan manuscript — v3 (2026-08-17)

This version is the second review-driven revision of the complete Chinese–English manuscript package. It preserves v1/v2 and resolves the remaining model-description and audit-scope issues identified in the v2 independent review. The intended journal is *Transportation Research Part C: Emerging Technologies*. Both manuscripts use Elsevier's `elsarticle` numeric-reference layout.

## Deliverables

- `en/main_en.tex` and `en/main_en.pdf`: English submission manuscript.
- `zh/main_zh.tex` and `zh/main_zh.pdf`: Chinese technical mirror used for checking meaning and data.
- `references.bib`: shared, DOI-checked bibliography.
- `figures/`: case-study, method, algorithm, result, ablation and sensitivity graphics.
- `tables/current_scheme_summary.csv`: machine-readable values used in the headline table.
- `tables/source_value_snapshot.csv`: 210 machine-extracted source values with evidence-scope labels; this is not an exhaustive parser-based LaTeX audit.
- `tables/source_manifest.csv`: hashes for result tables, current code and primary DEM/OSM/alignment inputs.
- `scripts/make_v1_figures.py`: read-only figure generator for the current experiment JSON.
- `DATA_AUDIT.md`: data lineage, scope boundaries and known limitations.
- `SUPPLEMENT_REPRODUCIBILITY.md`: full parameter/seed/code-variable map and command-level reproduction notes.
- `IMPLEMENTATION_METADATA_SIDECAR.json`: machine-readable errata for stale free-text metadata in the immutable result JSON and parameter documentation.
- `figures/IMAGEGEN_PROMPTS.md`: prompts and provenance for the two hand-drawn scientific illustrations.
- `review/`: independent review reports for this version.

## Authoritative data lineage

The primary current scheme is read from:

`../../优化方案对比（平面、纵断面联合协同优化）/results/joint_results_w500_dens.json`

It is the endpoint-anchored, building-density-enabled result with spectral width parameter `W=500 m`. `W` bounds the sine coefficients and is not a pointwise hard corridor. Older pre-density files are used only for explicitly labelled historical exploratory comparisons. They are not presented as the current scheme.

## Reproduce figures and PDFs

From this directory:

```bash
python3 scripts/make_v1_figures.py
python3 scripts/make_case_data_layers.py
python3 scripts/make_historical_spectral_plot.py
python3 scripts/audit_paper_numbers.py  # source extraction + five headline assertions
(cd en && tectonic main_en.tex)
(cd zh && tectonic main_zh.tex)
```

Tested with Tectonic on macOS. The generated PDFs are 29 pages (English) and 22 pages (Chinese). Minor `lineno.sty` UTF-8 and underfull-box warnings originate from the template/package stack and do not affect visible output.

## Template sources

- Elsevier LaTeX instructions: <https://www.elsevier.com/en-gb/researcher/author/policies-and-guidelines/latex-instructions>
- `elsarticle` package: <https://ctan.org/pkg/elsarticle>
- Journal scope: <https://shop.elsevier.com/journals/transportation-research-part-c-emerging-technologies/0968-090X>

Author names, affiliations, acknowledgements, funding identifiers and a public data DOI remain intentional submission placeholders.

## Evidence status after independent review

The current endpoint-anchored, density-enabled M-A/M-C numbers are directly traceable to one saved run. Ablation, multi-algorithm and 237-point sensitivity experiments were performed during an earlier free-endpoint, pre-density development stage and with unequal NFE; v3 labels them exploratory and does not use them to claim current-model robustness or fair algorithm superiority. Confirmatory submission claims require current-model multi-seed and equal-NFE reruns. Author/funding/data-DOI placeholders also remain a technical-submission blocker until the user supplies them.
