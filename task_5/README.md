# Task 5 — CT→MR Registration (SyN) + Organ Mask Transfer

HaN-Seg dataset · head-and-neck CT/MR · case_01

---

## 1. Goal

TotalSegmentator produces organ-at-risk (OAR) segmentations from **CT**, but several
head-and-neck OARs cannot be segmented directly from **MR** (see §6). The aim of this
task is to **register the CT and MR of the same patient** using a gold-standard
deformable algorithm (ANTs **SyN**), so that CT-based information and the MR can be
brought into a common coordinate space and the organ masks can be transferred/overlaid
for verification.

This task builds on an earlier rigid-registration attempt (SimpleITK + Mattes Mutual
Information) and replaces it with a proper resampling + SyN pipeline, together with
quantitative and visual evaluation.

---

## 2. Data

| Image | Matrix (voxels) | Spacing (mm) | Notes |
|-------|-----------------|--------------|-------|
| CT    | 1024 × 1024 × 202 | 0.558 × 0.558 × 2.0 | finer resolution, **larger field of view** |
| MR-T1 | 512 × 512 × 83    | 0.703 × 0.703 × 3.0 | coarser, **smaller field of view** |

Reference OAR masks are provided by HaN-Seg **in the CT coordinate system**
(see §5, "HaN-Seg caveat").

---

## 3. Pipeline

1. **Segment CT** with TotalSegmentator (head-and-neck subtasks) → organ masks in CT space.
2. **Resample both** CT and MR to a **common isotropic resolution** (1.0 mm) so they share a
   grid before registration.
3. **Register** with ANTs **SyNRA** (rigid → affine → SyN deformable), using **Mattes
   Mutual Information** as the metric (required because CT and MR have different intensity
   scales / are multimodal).
4. **Transfer masks** using nearest-neighbour interpolation (labels must never be
   linearly interpolated).
5. **Evaluate**:
   - global **Mutual Information** before vs after registration,
   - **per-structure alignment** overlays (mask outline on CT vs on registered MR),
   - checkerboard / fusion / edge-overlay QC figures.

### Fixed vs Moving decision

The **fixed** image defines the output coordinate space; the **moving** image is warped
onto it.

- This task uses **CT = fixed, MR = moving** (per supervisor instruction). Output lands in
  **CT space**, which is the standard convention in radiotherapy planning.
- Note: if the goal were "masks usable *on the MRI*", the correct choice would be
  **MR = fixed** instead, so that the masks land in MR space. The pipeline exposes this as a
  single toggle (`FIXED_MODALITY`) so the direction can be flipped without other changes.

---

## 4. Results (case_01)

**Global mutual information (CT ↔ MR):**

| Stage | Mutual Information |
|-------|--------------------|
| Before registration | 0.0000 |
| After registration (SyN) | 0.1055 |
| Change | **+0.1055** |

Interpretation: before registration the original CT and MR are **not in a shared world
coordinate frame**, so they essentially do not overlap and MI ≈ 0. After SyN they are
aligned and MI becomes clearly positive. This shows the registration was **necessary and
effective**. (MI is unitless and only meaningful *relatively* — before vs after — not as an
absolute "good/bad" threshold; see §5.)

**Per-structure visual alignment** (mask outline traced on CT vs on registered MR):

| Structure | Assessment |
|-----------|------------|
| **Spinal cord** | Strong — outline follows the cord cleanly on both CT and MR. |
| **Mandible** | Strong — bony outline lands on corresponding structure in MR. |
| **Parotid L / R** | Plausible — sits in the correct region; soft tissue has no sharp edge, so cannot be verified to sub-mm by eye (this is expected, not a defect). |

Figures are in `results/registration_syn/results/`:
`checkerboard_before_after.png`, `fusion_overlay.png`, `edge_overlay.png`,
`mutual_information.png` / `.csv`, and per-structure `structures/*_alignment.png` +
`overview_all_structures.png`.

---

## 5. Problems faced & how they were handled (full, honest account)

**1. "Fixed" vs "reference" vs "moving" terminology.**
Early on these were used inconsistently. Clarified: *fixed = reference = the image that stays
still and defines the output grid*; *moving = the image that gets warped*. The final choice
(CT fixed) is documented above with its consequence (output in CT space).

**2. Resolution mismatch.**
CT and MR have different voxel sizes and matrix sizes. Handled by resampling **both** to a
common 1.0 mm isotropic grid before registration. Note: pre-resampling is not strictly
required for correctness (registration works in physical space via the image affines and
uses an internal multi-resolution pyramid), but a common grid was used here for clarity and
because it was part of the task specification.

**3. Field-of-view mismatch (structures cut off).**
The MR has a **smaller field of view** than the CT. Structures lying outside the MR FOV
(e.g. **frontal sinus, zygomatic arch** in the earlier attempt) simply have no MR data to
map onto and appear cut off. **This is expected and unavoidable**, not a registration bug —
a mask is only meaningful where MR data exists.

**4. HaN-Seg caveat (important).**
Per the HaN-Seg dataset paper, each **MR was first registered to the CT** and the OARs were
then annotated **in the CT reference coordinate system**. Consequences:
- Ground-truth masks are in **CT space**, not MR space.
- Any registration evaluation that relies on the OAR GT is partly measuring **segmentation
  accuracy**, and the CT/MR pair is already partly co-registered by the dataset authors —
  so registration metrics on this dataset must be interpreted carefully.

**5. Evaluation-logic bug (found and corrected).**
An earlier evaluation applied the **registration transform to the prediction** but an
**identity transform to the ground truth**, which is inconsistent — it only yields a
sensible number if the transform is near-identity. The fair approach is to move both
prediction and GT into the same space with the **same** mapping, or to evaluate
segmentation directly in **CT space**. Flagged and corrected.

**6. Mask interpolation.**
Verified that all label/mask resampling uses **nearest-neighbour** interpolation. Linear
interpolation on a label image produces meaningless fractional values and would corrupt any
Dice score. Intensity images (CT, MR) correctly use linear interpolation.

**7. `MI before = 0.0000` could look like a failure.**
It is not. It reflects that the raw CT and MR do not share a world coordinate frame, so they
do not overlap prior to registration. This is the correct explanation to give if asked.

**8. Tooling choice — FreeSurfer was considered and rejected.**
FreeSurfer's registration tools (`bbregister`, surface pipeline) are **brain-specific** and
inappropriate for head-and-neck CT/MR. **ANTs SyN** was chosen instead as the field-standard
multimodal deformable method.

---

## 6. TotalSegmentator MR note

TotalSegmentator (v2.14.0) **does** support MR via `--task total_mr` and other `_mr` tasks.
However, the head-and-neck OARs used here (mandible, parotids, sinuses, nasal cavity,
zygomatic arch) come from **CT-only** subtasks (`head_glands_cavities`,
`headneck_bones_vessels`, etc.) that have **no MR equivalent**. Therefore MR-direct
segmentation does not replace registration for these structures, and CT→MR registration
remains necessary.

---

## 7. Suggestions / future work

- **Add a hard per-structure number:** compute **Dice between the TotalSegmentator mask and
  the HaN-Seg ground-truth mask in CT space** (no registration involved) so each organ has a
  quantitative segmentation-accuracy score alongside its picture.
- **Independent registration validation:** segment an MR-native structure (e.g. spinal cord
  via `total_mr`, or a vertebra via `vertebrae_mr`) directly in MR and compare against the
  registered-from-CT version — this tests the transform without circularity.
- **Affine vs SyN comparison:** rigid/affine is sufficient for bone (mandible); soft-tissue
  organs (parotids) may benefit from the deformable SyN component — quantify the difference.
- **Confirm output-space requirement** (CT space vs MR space) with the supervisor, since it
  determines the fixed/moving direction.
- **Scale to all 56 cases** once case_01 is validated, and report aggregate metrics.

---

## 8. Repository structure

```
task5_ct_mr_registration/
├── README.md
├── scripts/
│   ├── register_ct_mr_syn.py       # resample + SyN registration + mask transfer
│   ├── registration_results.py     # MI before/after + checkerboard/fusion/edge QC
│   └── structure_alignment.py      # per-structure outline on CT vs registered MR
├── results/
│   └── registration_syn/
│       └── results/
│           ├── checkerboard_before_after.png
│           ├── fusion_overlay.png
│           ├── edge_overlay.png
│           ├── mutual_information.png / .csv
│           └── structures/
│               ├── overview_all_structures.png
│               └── <Structure>_alignment.png
└── .gitignore                      # excludes large .nii.gz / .nrrd volumes
```

> Large image volumes (`.nii.gz`, `.nrrd`) are **not** committed to git (see `.gitignore`).
> Only scripts, README, and result figures/CSVs are versioned.

---

## 9. How to run

```bash
# environment
conda activate totalseg_env
pip install antspyx scipy          # one-time

# 1. registration (edit FIXED_MODALITY / COMMON_SPACING at top if needed)
python3 scripts/register_ct_mr_syn.py

# 2. quantitative + QC figures
python3 scripts/registration_results.py

# 3. per-structure alignment figures
python3 scripts/structure_alignment.py
```

Run from the case folder containing `case_01_IMG_CT.nii.gz`,
`case_01_IMG_MR_T1.nrrd`, and `case_01_seg/`.

---

## 10. Dependencies

- Python 3.10
- `antspyx` (ANTs SyN registration)
- `SimpleITK`
- `numpy`, `scipy`, `matplotlib`
- `TotalSegmentator` 2.14.0 (for the CT segmentation step)
