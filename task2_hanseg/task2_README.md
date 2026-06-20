# Task 2: HaN-Seg Head & Neck Segmentation

Pretrained-model segmentation of head and neck organs-at-risk (OARs), evaluated
against the HaN-Seg dataset. This is the second task in this repo, following
the same evaluate-a-pretrained-model-against-ground-truth approach as Task 1
(OASIS brain segmentation with SynthSeg).

**Status: pilot test on a single case (case_01). Full-dataset run pending supervisor sign-off.**

## Dataset

[HaN-Seg](https://doi.org/10.5281/zenodo.7442914) (Zenodo, CC-BY-NC-ND 4.0) — 56
patients (42 train / 14 test), each with:
- A CT scan
- A T1-weighted MRI scan (registered to the CT)
- 30 manually-segmented OAR masks (e.g. Parotid, Brainstem, Cochlea, SpinalCord,
  Carotid, Larynx_SG), one `.nrrd` file per organ

case_01 (used for this pilot test): CT/MR at 1024×1024×202 voxels,
0.558×0.558×2.0mm spacing.

## Why CT, not MRI

The task brief was to segment using the MRI. Two pretrained options were
evaluated and ruled out for MRI:

- **SynthSeg** — works well on MRI (see Task 1), but is brain-only. It has no
  concept of parotid glands, larynx, thyroid, etc., so it cannot address
  HaN-Seg's organ set regardless of modality.
- **TotalSegmentator** — has by far the best head/neck organ coverage of any
  public pretrained model, but is trained purely on CT. CT intensities are
  calibrated, physically-meaningful Hounsfield Units; MRI intensities are
  scanner- and scan-relative with no fixed scale. A model trained on HU values
  cannot be applied to MRI data. TotalSegmentator does have a separate `total_mr`
  model, but it only covers ~50 abdominal/pelvic/musculoskeletal structures —
  the only head-related label is whole "brain," nothing finer.

No publicly available pretrained model performs head/neck OAR segmentation on
MRI directly. Given this, the pivot was to test TotalSegmentator on the CT
scan that ships alongside each HaN-Seg case, as a CT-based reference baseline,
while the MRI-native approach (most likely training nnU-Net from scratch on
HaN-Seg's own 42 labeled training cases) is decided on separately.

## Method

1. Converted `case_01_IMG_CT.nrrd` → `.nii.gz` (SimpleITK), since TotalSegmentator
   expects NIfTI input.
2. Ran TotalSegmentator (CT, pretrained, no training) across its three
   head/neck-relevant task groups: `total`, `head_glands_cavities`,
   `headneck_bones_vessels` — 148 organ mask files produced.
3. Built an explicit name-mapping table between TotalSegmentator's output
   classes and HaN-Seg's 30 ground-truth OAR names (different naming
   conventions; see `scripts/totalseg_results.py`).
4. Computed per-organ Dice (DSC) for every HaN-Seg OAR that has a
   TotalSegmentator equivalent.

## Coverage

Of HaN-Seg's 30 OARs, only **11** have any equivalent in TotalSegmentator's
vocabulary at all — this is a hard ceiling, independent of prediction
accuracy.

| Covered (11) | Not covered (19) |
|---|---|
| Parotid L/R, Submandibular L/R, Optic Nerve L/R, Spinal Cord, Thyroid, Carotid L/R, Esophagus | Brainstem, OpticChiasm, Cochlea L/R, Pituitary, Bone_Mandible, Arytenoid, Cricopharyngeus, Glottis, Larynx_SG, Lips, Cavity_Oral, BuccalMucosa, Glnd_Lacrimal L/R, Eye_AL/AR/PL/PR |

![Coverage Chart](results/totalseg_coverage_chart.png)

## Results — case_01

Mean DSC across the 11 matched structures: **0.562**

| Structure | DSC | Note |
|---|---|---|
| Submandibular L | 0.832 | |
| Submandibular R | 0.825 | |
| Thyroid | 0.821 | |
| Carotid R | 0.668 | approx: union of TotalSegmentator's common + internal carotid classes |
| Carotid L | 0.653 | approx: union of TotalSegmentator's common + internal carotid classes |
| Parotid L | 0.612 | |
| Optic Nerve R | 0.547 | |
| Spinal Cord | 0.391 | lower than expected for a well-defined structure — flagged for follow-up |
| Parotid R | 0.390 | |
| Optic Nerve L | 0.336 | lower than expected for a well-defined structure — flagged for follow-up |
| Esophagus (S) | 0.110 | not a fair comparison: TotalSegmentator segments the full esophagus, HaN-Seg ground truth only the cervical portion |

![Dice Chart](results/totalseg_dice_chart.png)
![Overlay Comparison](results/totalseg_overlay_comparison.png)

## Limitations

- **Single-case pilot only.** These numbers are from case_01 alone, not yet
  averaged across the dataset.
- **Coverage ceiling.** Even with perfect predictions, TotalSegmentator can
  never address more than 11/30 HaN-Seg OARs.
- **Two approximate matches.** Carotid (union of two TotalSegmentator vessel
  classes) and Esophagus (region-definition mismatch) are not clean 1-to-1
  comparisons — see notes above.
- Run on CPU (GPU driver mismatch on the compute server); full-dataset runtime
  will need this resolved first.

## Next steps

Pending decision: scale this CT/TotalSegmentator pipeline to all 56 cases as a
partial (11/30 organ) baseline, or move to training nnU-Net from scratch on
HaN-Seg's own labeled training cases for full 30-organ coverage on MRI.

## Repository structure

```
task2_hanseg_segmentation/
├── scripts/
│   └── totalseg_results.py      # Dice scoring + visualization
├── results/
│   ├── totalseg_dice_results.csv
│   ├── totalseg_dice_chart.png
│   ├── totalseg_coverage_chart.png
│   └── totalseg_overlay_comparison.png
└── task2_README.md
```
