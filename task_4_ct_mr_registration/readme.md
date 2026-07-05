# Task 2: HaN-Seg Head & Neck Segmentation — CT→MR Registration

## Overview
TotalSegmentator was run on the HaN-Seg case_01 CT scan to produce organ segmentation masks. Since TotalSegmentator only works on CT, rigid registration was performed to transfer these masks into MRI space for downstream MRI-based evaluation.

## Dataset
[HaN-Seg](https://doi.org/10.5281/zenodo.7442914) (Zenodo, CC-BY-NC-ND 4.0) — 56 patients (42 train / 14 test), each with a CT scan, T1 MRI, and 30 manually annotated OAR masks.

**case_01 scan properties:**
| Modality | Size | Spacing |
|---|---|---|
| CT | 1024×1024×202 | 0.558×0.558×2.0 mm |
| MRI (T1) | 512×512×83 | 0.703×0.703×3.0 mm |

## Method

**Step 1 — TotalSegmentator on CT**
Ran TotalSegmentator (pretrained, no training) across multiple head/neck task groups: `total`, `head_glands_cavities`, `headneck_bones_vessels`, `craniofacial_structures`. Produced 148 organ mask files.

**Step 2 — Rigid CT→MR Registration**
- Fixed image: MRI (stays still)
- Moving image: CT (transformed to match MRI)
- Transform: Rigid — 6 parameters (3 Euler angle rotations + 3 translations)
- Metric: Mattes Mutual Information — handles CT/MRI intensity scale mismatch by measuring statistical correspondence rather than direct pixel comparison
- Optimizer: Gradient Descent, 200 iterations
- Strategy: 3-level multiresolution (shrink factors 4→2→1, smoothing sigmas 2→1→0)
- Interpolation: Linear for CT image, Nearest Neighbour for binary masks

**Step 3 — Evaluation**
- Mutual Information after registration as overall alignment metric
- Dice score for structures with available HaN-Seg ground truth
- 4-panel visual verification per structure

## Focus Structures
| Structure | TotalSegmentator file | HaN-Seg GT |
|---|---|---|
| Mandible | mandible.nii.gz | Yes |
| Maxillary Sinus | sinus_maxillary.nii.gz | No |
| Nasal Cavity | nasal_cavity_left/right.nii.gz | No |
| Parotid L | parotid_gland_left.nii.gz | Yes |
| Parotid R | parotid_gland_right.nii.gz | Yes |
| Spinal Cord | spinal_cord.nii.gz | Yes |

## Results

**Registration quality:**
- Mattes Mutual Information after registration: **-0.451**

**Dice scores (registered mask vs HaN-Seg ground truth):**
| Structure | DSC | Note |
|---|---|---|
| Mandible | **0.9268** | Excellent — mask correctly aligned to jaw anatomy on MRI |
| Maxillary Sinus | N/A | No HaN-Seg GT — visually correct placement |
| Nasal Cavity | N/A | No HaN-Seg GT — mostly correct, slight FOV cutoff |
| Parotid L/R | N/A | Partial — voxel loss due to MRI FOV mismatch |
| Spinal Cord | N/A | Partial — cord extends below MRI coverage boundary |

**Field of view mismatch:** MRI covers a smaller physical region than CT (83 slices vs 202 slices). Structures near the edges of the head (Frontal Sinus, Zygomatic Arch) fell entirely outside MRI coverage after registration — this is a dataset limitation, not a registration error.

## Visualizations
Each structure has a 4-panel figure:
1. CT + mask (TotalSegmentator prediction in CT space)
2. MRI alone (reference)
3. MRI + registered mask (after CT→MR registration)
4. Registered CT + mask (alignment check — both in MRI space)

## Next Steps
- Resample CT and MRI to same resolution before registration to handle voxel spacing mismatch
- Investigate ANTs (Advanced Normalization Tools) as the standard registration algorithm
- Reconsider fixed/moving image convention based on use case
- Scale pipeline to all 56 HaN-Seg cases

## Repository Structure
```
task2_hanseg_segmentation/
├── scripts/
│   ├── ct_to_mr_registration.py        ← runs registration, saves transform
│   ├── registration_eval_v2.py         ← MI + Dice evaluation + visualizations
│   └── registration_viz_proper.py      ← 4-panel per-structure figures
├── results/
│   ├── registration_eval/
│   │   ├── MI_registration_metric.csv
│   │   ├── dice_results.csv
│   │   └── *.png                       ← three-plane overlays
│   └── registration_viz_proper/
│       ├── summary_all_structures.png
│       └── *_registration.png          ← 4-panel per structure
└── README.md
```
