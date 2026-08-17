# RoadPlan manuscript — v1 (2026-08-16)

This immutable version contains a complete Chinese–English manuscript package derived from the current RoadPlan experiments. The intended journal is *Transportation Research Part C: Emerging Technologies*. Both manuscripts use Elsevier's `elsarticle` numeric-reference layout.

## Deliverables

- `en/main_en.tex` and `en/main_en.pdf`: English submission manuscript.
- `zh/main_zh.tex` and `zh/main_zh.pdf`: Chinese technical mirror used for checking meaning and data.
- `references.bib`: shared, DOI-checked bibliography.
- `figures/`: case-study, method, algorithm, result, ablation and sensitivity graphics.
- `tables/current_scheme_summary.csv`: machine-readable values used in the headline table.
- `scripts/make_v1_figures.py`: read-only figure generator for the current experiment JSON.
- `DATA_AUDIT.md`: data lineage, scope boundaries and known limitations.
- `figures/IMAGEGEN_PROMPTS.md`: prompts and provenance for the two hand-drawn scientific illustrations.
- `review/`: independent review reports for this version.

## Authoritative data lineage

The primary current scheme is read from:

`../../优化方案对比（平面、纵断面联合协同优化）/results/joint_results_w500_dens.json`

It is the endpoint-anchored, building-density-enabled, ±500 m result. Older pre-density files are used only for the explicitly labelled controlled joint-versus-two-stage comparison and algorithm experiments. They are not presented as the current scheme.

## Reproduce figures and PDFs

From this directory:

```bash
python3 scripts/make_v1_figures.py
(cd en && tectonic main_en.tex)
(cd zh && tectonic main_zh.tex)
```

Tested with Tectonic on macOS. The generated PDFs are 25 pages (English) and 20 pages (Chinese). Minor `lineno.sty` UTF-8 and underfull-box warnings originate from the template/package stack and do not affect visible output.

## Template sources

- Elsevier LaTeX instructions: <https://www.elsevier.com/en-gb/researcher/author/policies-and-guidelines/latex-instructions>
- `elsarticle` package: <https://ctan.org/pkg/elsarticle>
- Journal scope: <https://shop.elsevier.com/journals/transportation-research-part-c-emerging-technologies/0968-090X>

Author names, affiliations, acknowledgements, funding identifiers and a public data DOI remain intentional submission placeholders.
