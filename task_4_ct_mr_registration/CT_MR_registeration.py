"""
CT to MR Registration — case_01 HaN-Seg
========================================
Goal: align the CT scan (and TotalSegmentator masks on it) to the MR scan
of the same patient, so the masks end up in MR space.

What this script does step by step:
1. Load CT and MR
2. Resample MR to same grid as CT (so they can be compared)
3. Run rigid registration (CT moves to match MR)
4. Save the transform
5. Apply transform to 4 focus organ masks
6. Save the registered masks in MR space
7. Generate a before/after visualization to verify it worked

Run from inside hanseg_case01 folder:
    conda activate viz_env
    python3 ct_to_mr_registration.py

Author: generated for HaN-Seg Task 2
"""

import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─── PATHS ────────────────────────────────────────────────────────────────
CASE_DIR  = "."
CT_PATH   = os.path.join(CASE_DIR, "case_01_IMG_CT.nii.gz")
MR_PATH   = os.path.join(CASE_DIR, "case_01_IMG_MR_T1.nrrd")
SEG_DIR   = os.path.join(CASE_DIR, "case_01_seg")
OUT_DIR   = os.path.join(CASE_DIR, "registration_results")
os.makedirs(OUT_DIR, exist_ok=True)

# The 4 focus structures we care about
FOCUS_MASKS = [
    ("Parotid_L",   "parotid_gland_left.nii.gz",  "#4a90d9"),
    ("Parotid_R",   "parotid_gland_right.nii.gz",  "#e87040"),
    ("Spinal_Cord", "spinal_cord.nii.gz",           "#5bc8af"),
    ("Mandible",    "mandible.nii.gz",               "#e8c840"),
]

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────
def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def norm(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn: return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)

def make_overlay(ct_slice, masks_and_colors, alpha=0.6):
    rgb = np.stack([norm(ct_slice)]*3, axis=-1)
    for mask2d, color in masks_and_colors:
        if not mask2d.any(): continue
        rgb[mask2d] = (1-alpha)*rgb[mask2d] + alpha*np.array(color)
    return rgb

# ─── STEP 1: LOAD IMAGES ──────────────────────────────────────────────────
print("="*55)
print("STEP 1: Loading CT and MR images")
print("="*55)

ct_sitk = sitk.ReadImage(CT_PATH, sitk.sitkFloat32)
mr_sitk = sitk.ReadImage(MR_PATH, sitk.sitkFloat32)

print(f"  CT  size: {ct_sitk.GetSize()}  spacing: {[round(s,3) for s in ct_sitk.GetSpacing()]}")
print(f"  MR  size: {mr_sitk.GetSize()}  spacing: {[round(s,3) for s in mr_sitk.GetSpacing()]}")

# ─── STEP 2: INITIAL ALIGNMENT ────────────────────────────────────────────
# Before running the full optimization, we do a quick initial alignment
# by matching the centers of mass of both images.
# This gives the optimizer a good starting point so it doesn't get stuck.
print("\n" + "="*55)
print("STEP 2: Initial alignment (center of mass)")
print("="*55)

initial_transform = sitk.CenteredTransformInitializer(
    mr_sitk,                          # fixed  = MR (stays still)
    ct_sitk,                          # moving = CT (will be transformed)
    sitk.Euler3DTransform(),          # rigid transform: 3 rotations + 3 translations
    sitk.CenteredTransformInitializerFilter.GEOMETRY  # align geometric centers
)
print("  Initial transform computed (centers aligned)")

# ─── STEP 3: SET UP REGISTRATION ──────────────────────────────────────────
# This is the core of registration. We define:
#   - Metric:    HOW to measure similarity between CT and MR
#   - Optimizer: HOW to search for the best transform
#   - Transform: WHAT kind of transformation to allow (rigid = shift + rotate only)
#   - Interpolator: HOW to compute pixel values at non-integer positions
print("\n" + "="*55)
print("STEP 3: Setting up registration framework")
print("="*55)

registration = sitk.ImageRegistrationMethod()

# METRIC: Mattes Mutual Information
# Works across different modalities (CT vs MR) because it doesn't
# compare raw pixel values — it measures statistical dependency between them.
# numberOfHistogramBins=50 means it buckets intensity values into 50 bins.
registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

# Only sample a fraction of voxels for speed (0.01 = 1% of voxels per iteration)
# This is enough for a reliable estimate and makes it much faster.
registration.SetMetricSamplingStrategy(registration.RANDOM)
registration.SetMetricSamplingPercentage(0.01)

# INTERPOLATOR: Linear interpolation
# When the optimizer moves/rotates the CT slightly, most positions won't
# land exactly on a voxel center. Linear interpolation estimates the
# intensity at those in-between positions.
registration.SetInterpolator(sitk.sitkLinear)

# OPTIMIZER: Gradient Descent
# Starts from the initial transform and iteratively adjusts it to
# increase the mutual information metric.
# learningRate: how big each step is
# numberOfIterations: how many steps to take
# convergenceMinimumValue: stop early if improvement is tiny
registration.SetOptimizerAsGradientDescent(
    learningRate=1.0,
    numberOfIterations=200,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)
# This scales the optimizer so rotation and translation steps are comparable
registration.SetOptimizerScalesFromPhysicalShift()

# Use multiple resolutions (coarse → fine) for speed and robustness.
# Starts at 1/4 resolution, then 1/2, then full resolution.
# This is called "multi-resolution" or "pyramid" registration.
registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Set the initial transform we computed in Step 2
registration.SetInitialTransform(initial_transform, inPlace=False)

print("  Registration configured:")
print("  - Metric: Mattes Mutual Information (works for CT↔MR)")
print("  - Transform: Rigid (3 rotations + 3 translations)")
print("  - Optimizer: Gradient Descent (200 iterations)")
print("  - Multi-resolution: 3 levels (coarse → fine)")

# ─── STEP 4: RUN REGISTRATION ─────────────────────────────────────────────
print("\n" + "="*55)
print("STEP 4: Running registration (this takes a few minutes...)")
print("="*55)

# Optional: print metric value every 10 iterations so you can watch progress
def iteration_callback(method):
    if method.GetOptimizerIteration() % 10 == 0:
        print(f"  iter {method.GetOptimizerIteration():3d} | "
              f"metric = {method.GetMetricValue():.6f} | "
              f"position = {[round(x,3) for x in method.GetOptimizerPosition()[:3]]}")

registration.AddCommand(sitk.sitkIterationEvent,
                        lambda: iteration_callback(registration))

final_transform = registration.Execute(
    sitk.Cast(mr_sitk, sitk.sitkFloat32),   # fixed  = MR
    sitk.Cast(ct_sitk, sitk.sitkFloat32)    # moving = CT
)

print(f"\n  Registration complete!")
print(f"  Final metric value: {registration.GetMetricValue():.6f}")
print(f"  Optimizer stopped: {registration.GetOptimizerStopConditionDescription()}")

# Save the transform so you can reuse it without re-running registration
transform_path = os.path.join(OUT_DIR, "ct_to_mr_rigid_transform.tfm")
sitk.WriteTransform(final_transform, transform_path)
print(f"  Transform saved: {transform_path}")

# ─── STEP 5: APPLY TRANSFORM TO CT ────────────────────────────────────────
# Resample the CT into MR space using the transform we just found.
# This produces a CT image that is aligned with the MR.
print("\n" + "="*55)
print("STEP 5: Applying transform to CT")
print("="*55)

ct_registered = sitk.Resample(
    ct_sitk,            # image to transform
    mr_sitk,            # use MR as reference (defines output grid)
    final_transform,    # transform to apply
    sitk.sitkLinear,    # interpolator (linear for CT intensity image)
    -1000.0,            # default value for voxels outside CT field of view (air in HU)
    ct_sitk.GetPixelID()
)

ct_reg_path = os.path.join(OUT_DIR, "case_01_CT_registered_to_MR.nii.gz")
sitk.WriteImage(ct_registered, ct_reg_path)
print(f"  Registered CT saved: {ct_reg_path}")

# ─── STEP 6: APPLY TRANSFORM TO ORGAN MASKS ───────────────────────────────
# Same transform, applied to each binary mask.
# IMPORTANT: use sitkNearestNeighbor interpolation for masks,
# NOT linear — because masks are binary (0 or 1) and linear
# interpolation would create weird intermediate values like 0.3 or 0.7.
print("\n" + "="*55)
print("STEP 6: Applying transform to organ masks")
print("="*55)

registered_masks = {}
for name, fname, color_hex in FOCUS_MASKS:
    mask_path = os.path.join(SEG_DIR, fname)
    if not os.path.exists(mask_path):
        print(f"  WARNING: {fname} not found — skipping {name}")
        continue

    mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)

    mask_registered = sitk.Resample(
        mask_sitk,
        mr_sitk,                        # reference = MR space
        final_transform,
        sitk.sitkNearestNeighbor,       # nearest neighbor for binary masks!
        0,                              # default = 0 (background)
        mask_sitk.GetPixelID()
    )

    out_mask_path = os.path.join(OUT_DIR, f"{name}_registered_to_MR.nii.gz")
    sitk.WriteImage(mask_registered, out_mask_path)
    registered_masks[name] = (sitk.GetArrayFromImage(mask_registered) > 0, color_hex)
    print(f"  {name}: saved → {out_mask_path}")

# ─── STEP 7: VISUALIZATION — BEFORE / AFTER ───────────────────────────────
print("\n" + "="*55)
print("STEP 7: Generating before/after visualization")
print("="*55)

mr_arr  = sitk.GetArrayFromImage(mr_sitk)        # z,y,x
ct_arr  = sitk.GetArrayFromImage(ct_sitk)        # z,y,x  (original, unregistered)
ct_reg_arr = sitk.GetArrayFromImage(ct_registered) # z,y,x (registered)

# Find best axial slice (most mask content)
combined = np.zeros(mr_arr.shape, dtype=bool)
for name, (mask_arr, _) in registered_masks.items():
    if mask_arr.shape == mr_arr.shape:
        combined |= mask_arr
best_z = int(np.argmax(combined.sum(axis=(1,2))))
print(f"  Using axial slice z={best_z} for visualization")

masks_and_colors = [
    (mask_arr[best_z], hex_to_rgb(color_hex))
    for name, (mask_arr, color_hex) in registered_masks.items()
    if mask_arr.shape == mr_arr.shape
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#0d0d0d')
fig.suptitle(f'CT→MR Registration — case_01  (axial z={best_z})',
             color='white', fontsize=13, fontweight='bold')

titles = ['MR (fixed)', 'CT (before reg)', 'CT (after reg)', 'MR + registered masks']
images = [
    norm(mr_arr[best_z]),
    norm(ct_arr[best_z]) if best_z < ct_arr.shape[0] else np.zeros_like(mr_arr[best_z]),
    norm(ct_reg_arr[best_z]),
    make_overlay(mr_arr[best_z], masks_and_colors)
]
cmaps = ['gray', 'gray', 'gray', None]

for ax, title, img, cmap in zip(axes, titles, images, cmaps):
    if cmap:
        ax.imshow(img, cmap=cmap, origin='lower', aspect='auto')
    else:
        ax.imshow(img, origin='lower', aspect='auto')
    ax.set_title(title, color='white', fontsize=11)
    ax.axis('off')
    ax.set_facecolor('black')

# Legend for the masks panel
patches = [mpatches.Patch(color=hex_to_rgb(c), label=n)
           for n, (_, c) in registered_masks.items()]
axes[3].legend(handles=patches, loc='lower right', facecolor='#1a1a1a',
               edgecolor='#444', labelcolor='white', fontsize=9)

plt.tight_layout()
out_fig = os.path.join(OUT_DIR, "registration_before_after.png")
plt.savefig(out_fig, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"  Saved: {out_fig}")

# ─── SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("REGISTRATION COMPLETE")
print("="*55)
print(f"All outputs saved to: {OUT_DIR}/")
print("  ct_to_mr_rigid_transform.tfm       ← the transform (reusable)")
print("  case_01_CT_registered_to_MR.nii.gz ← CT in MR space")
print("  Parotid_L_registered_to_MR.nii.gz")
print("  Parotid_R_registered_to_MR.nii.gz")
print("  Spinal_Cord_registered_to_MR.nii.gz")
print("  Mandible_registered_to_MR.nii.gz")
print("  registration_before_after.png       ← visual verification")
print("\nNext step: compute Dice between registered masks and")
print("HaN-Seg ground truth OAR masks to measure accuracy.")