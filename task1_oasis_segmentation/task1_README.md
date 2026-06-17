# Task 1 — Brain MRI Segmentation on OASIS Dataset

## Objective

Perform multi-structure brain MRI segmentation using the OASIS dataset (Learn2Reg challenge version) using a pretrained model, SynthSeg.

## Dataset

**OASIS (Open Access Series of Imaging Studies)** — Learn2Reg version

- 416 total subjects (394 training, 22 test)
- Modality: T1-weighted MRI
- Image size: 160 × 224 × 192 voxels, 1mm isotropic spacing
- 35 labeled brain structures (consecutive IDs 1–35), following FreeSurfer anatomical conventions
- Source: https://learn2reg.grand-challenge.org/Datasets/

For this task, 5 sample subjects were used: OASIS_0006, OASIS_0010, OASIS_0019, OASIS_0026, OASIS_0034.

## Models Used

### 1. SynthSeg (pretrained, no training required)

SynthSeg is a domain-randomized U-Net trained entirely on synthetic data, designed to generalize to any MRI contrast and resolution without retraining. It outputs 33 FreeSurfer-style brain structure labels.

- Repository: https://github.com/BBillot/SynthSeg
- Version used: SynthSeg 1.0 (`--v1` flag), Python 3.8, TensorFlow 2.2.0, Keras 2.3.1
- Run on: remote GPU server (NVIDIA RTX 6000 Ada, 49GB VRAM)

**Why SynthSeg and not an nnUNet pretrained checkpoint:** nnUNet does not publicly distribute pretrained brain segmentation weights (confirmed via open GitHub issue on MIC-DKFZ/nnUNet). SynthSeg is the standard openly available pretrained brain MRI segmentation model and is built into FreeSurfer.

## Label Mapping (critical step)

OASIS ground truth uses consecutive label IDs (1–35). SynthSeg outputs FreeSurfer-convention IDs (2, 3, 4, 5, 7, 8, 10...60). These do **not** match by default and must be explicitly remapped before computing Dice — otherwise Dice scores will be near-zero despite correct segmentation. The mapping was sourced from the official Learn2Reg OASIS documentation and verified against the dataset's published structure list.

See `scripts/synthseg_mapping.py` for the full mapping table.

## Evaluation Metric

**Dice Similarity Coefficient (DSC)**, computed per structure and averaged:

```
DSC = 2 * |Prediction ∩ GroundTruth| / (|Prediction| + |GroundTruth|)
```

## Results

**Mean DSC (SynthSeg, 31 structures, 5 subjects): 0.843** — consistent with published SynthSeg benchmarks.

### SynthSeg per-structure highlights
- Best: R Cerebral WM (0.922), L Cerebral WM (0.920), L Lateral Ventricle (0.912)
- Weakest: L Inferior Lateral Ventricle (0.515), R Inferior Lateral Ventricle (0.627) — smallest structures in the dataset, high inter-subject variability
- Not segmented by SynthSeg 1.0: vessels and choroid plexus (4 of 35 OASIS labels)

See `results/` for full per-structure breakdowns, three-plane overlays, axial slice comparisons, and error maps (true/false positive/negative visualization).

## Files

- `scripts/synthseg_mapping.py` — label remapping logic (OASIS consecutive IDs → FreeSurfer IDs)
- `scripts/synthseg_results.py` — Dice computation + all visualizations
- `results/` — generated figures (`*_three_plane.png`, `*_dice_chart.png`, `*_error_map.png`, `*_axial_slices.png`, `*_per_subject.png`) and `synthseg_dice_results.csv`

## Key Lessons / Notes for Reproduction

- SynthSeg requires Python 3.6 or 3.8 specifically (`setup.py` enforces this) — does not run on Python 3.10+
- Exact dependency versions matter a lot: `tensorflow-gpu==2.2.0 keras==2.3.1 protobuf==3.20.3 numpy==1.23.5 nibabel==5.0.1 matplotlib==3.6.2` for the Python 3.8 path
- Label ID mismatch between dataset ground truth and model output convention is a common but easy-to-miss source of artificially low Dice scores — always verify `np.unique()` on both volumes before evaluating
