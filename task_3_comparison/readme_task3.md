# Task 3: MIDA vs TotalSegmentator — Anatomical Structure Comparison

## Overview
A systematic comparison of anatomical structures covered by the MIDA atlas and TotalSegmentator, organized by anatomical region. Built to identify structural correspondences that can serve as registration anchors for atlas-based image reconstruction.

## Background
**MIDA** (Iacono et al. 2015) is a multimodal imaging-based detailed anatomical atlas of the human head and neck, with 153 hand-labeled structures from a single healthy volunteer, available as voxel and surface mesh models. It covers the head and neck down to cervical vertebra C5 only.

**TotalSegmentator** (Wasserthal et al. 2023) is a pretrained deep learning model that automatically segments 117+ anatomical structures from CT scans, with additional head/neck-specific task groups added post-publication.

The structure list for TotalSegmentator was extracted directly from the installed package (v2.14.0, `map_to_binary.py`) across 9 head/neck-relevant tasks: `total`, `head_glands_cavities`, `headneck_bones_vessels`, `head_muscles`, `headneck_muscles`, `brain_structures`, `craniofacial_structures`, `oculomotor_muscles`, `teeth`.

## Spreadsheet Structure
**Sheet 1 — Head & Neck overlap**

| Column | Description |
|---|---|
| Anatomical region | Body region (e.g. Head — Brainstem, Neck — Salivary glands) |
| TotalSegmentator structures | Exact class names from installed package v2.14.0 |
| MIDA structures | Structure names from MIDA Table 1 and Table 2 |
| Overlap? | Both / Partial / MIDA only / TS only |
| Exact matching structures (★) | Structure-level name mapping — TS name ↔ MIDA name. Orange highlighted rows are initial focus structures with clean 1-to-1 matches |

**Overlap categories:**
- 🟢 **Both** — clean match, usable as registration anchor
- 🟡 **Partial** — same anatomical region, different granularity (e.g. TS gives one brainstem label, MIDA gives midbrain + pons + medulla separately)
- 🔴 **MIDA only** — no TotalSegmentator equivalent (e.g. all 12 cranial nerves, facial expression muscles, dura mater)
- 🟣 **TS only** — below MIDA's C5 coverage boundary (e.g. thyroid gland, trachea, larynx cartilages)

**Sheet 2 — Legend & Notes:** sources, methodology, and caveats.

## Key Findings
- Clean 1-to-1 matches include: masseter, temporalis, medial/lateral pterygoid, parotid gland (L/R), submandibular gland (L/R), carotid arteries, cervical vertebrae C1–C5, spinal cord, mandible, tongue, extraocular muscles
- MIDA-only structures (no TS equivalent): all cranial nerves except optic nerve, facial expression muscles, dura mater, skull layers (outer table / diploe / inner table), skin
- No publicly available pretrained model covers all 153 MIDA structures automatically

## Sources
- MIDA: Iacono MI et al. (2015). *PLoS ONE* 10(4): e0124126. https://doi.org/10.1371/journal.pone.0124126
- TotalSegmentator: Wasserthal J et al. (2023). *Radiology: Artificial Intelligence* 5(5): e230024. https://doi.org/10.1148/ryai.230024

## Files
```
task3_mida_totalseg_comparison/
└── MIDA_vs_TotalSeg_FINAL.xlsx
```
