# Image-generation record

Mode: built-in OpenAI image generation, new-image workflow, no reference image. The outputs were visually inspected at full page scale and then copied into this version unchanged.

## `method_overview_handdrawn.png`

Prompt:

> Use case: scientific-educational. Asset type: a full-width methodology overview figure for an SCI transportation journal article. Create a rigorous landscape scientific diagram showing an end-to-end workflow for data-driven three-dimensional highway alignment optimization. Use five clearly separated left-to-right panels connected by bold arrows: (1) DATA with small DEM terrain, OSM road/water/building map, and GNSS centerline sketches; (2) DIGITAL CORRIDOR with a realistic mountain–city corridor, highway, bridge and tunnel; (3) COUPLED MODEL with plan-view horizontal alignment above a vertical profile, plus life-cycle-cost coins and mixed fuel/electric energy icons; (4) IJS with a jellyfish and three compact operator labels Tent, Lévy and DE; (5) OPTIMIZED ALIGNMENT with a feasible 3D highway, a small Pareto curve and validation check marks. Editorial hand-drawn scientific illustration, fine ink outlines, colored-pencil/watercolor texture, warm off-white paper, restrained teal/blue/ochre/coral palette, technically credible rather than whimsical. Wide 16:9 composition, clean margins, high information density, readable short English labels only. No equations, no paragraphs, no logos, no watermark, no decorative title outside the panels.

Generated source: `/Users/cengwanmin/.codex/generated_images/01a0085f-8a74-7331-a3fa-d961d76df9f0/exec-b0236def-3f0b-4a00-9a1c-df5b23e58489.png`.

## `ijs_flowchart_handdrawn.png`

Prompt:

> Use case: scientific-educational. Asset type: portrait algorithm flowchart for an SCI transportation journal. Draw an exact, logically connected flowchart for Improved Jellyfish Search used in 3D highway alignment optimization. Sequence from top: START; INPUT DATA & PARAMETERS; ENTROPY WEIGHTS; TENT INITIALIZATION; EVALUATE LCC, ENERGY & CONSTRAINTS; decision diamond STOP? The YES branch goes right to PARETO SCREENING, then FEASIBLE KNEE SOLUTION, then OUTPUT 3D ALIGNMENT. The NO branch goes downward through OCEAN CURRENT / ACTIVE–PASSIVE MOTION; ELITIST ACCEPTANCE; LÉVY FLIGHT; ELITIST ACCEPTANCE; DE MUTATION & CROSSOVER; ELITIST ACCEPTANCE; UPDATE BEST & DIVERSITY; then a single loop arrow returns to STOP?. Use blue for data/decision stages, muted coral for search operators and green for the feasible selected solution. Editorial hand-drawn scientific-cartoon style, fine ink, subtle colored-pencil texture on warm off-white paper, neat rounded rectangles and arrows, highly legible short English labels, ample whitespace, portrait 4:5 layout. No equations, no extra branches, no logos, no watermark.

Generated source: `/Users/cengwanmin/.codex/generated_images/01a0085f-8a74-7331-a3fa-d961d76df9f0/exec-1e313351-4ec3-41f4-87bb-3ae5afeed032.png`.

### v2 terminology correction

The v1 flowchart incorrectly said “FEASIBLE KNEE SOLUTION”; the implementation selects an entropy-weighted compromise after screening. The v2 image was edited with this prompt:

> Edit this existing scientific hand-drawn flowchart with exactly one content correction. Replace the green box text “FEASIBLE KNEE SOLUTION” with “ENTROPY-WEIGHTED COMPROMISE”. Preserve every other label, arrow, box, branch, color, texture, dimensions, composition, typography style, and the warm off-white hand-drawn scientific-cartoon aesthetic unchanged. Do not add or remove any stage. Ensure the new green-box wording is fully legible and centered.

Edited source: `/Users/cengwanmin/.codex/generated_images/01a0085f-8a74-7331-a3fa-d961d76df9f0/exec-dd4247f4-29e7-4c5e-a1af-a127ca8826c0.png`.
