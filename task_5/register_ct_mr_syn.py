"""
CT-MR Registration with ANTs (resample to common resolution + SyN)
===================================================================
Does exactly three things:
  1. Resamples BOTH CT and MR to the same voxel resolution.
  2. Registers them with SyN (the gold-standard deformable algorithm).
  3. Warps the CT organ masks into the fixed image's space.

Install first (one time):
    pip install antspyx

Run:
    conda activate totalseg_env
    python3 register_ct_mr_syn.py
"""

import ants
import os

# ============================ CONFIG ============================
CASE_DIR = "."
CT_PATH  = os.path.join(CASE_DIR, "case_01_IMG_CT.nii.gz")
MR_PATH  = os.path.join(CASE_DIR, "case_01_IMG_MR_T1.nrrd")
SEG_DIR  = os.path.join(CASE_DIR, "case_01_seg")
OUT_DIR  = os.path.join(CASE_DIR, "registration_syn")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- THE ONE CHOICE THAT MATTERS ----
# The output ends up on the FIXED image's grid / space.
#   "MR" = masks end up in MR space   (matches "masks usable with MRI")
#   "CT" = everything ends up in CT space
FIXED_MODALITY = "CT"          # <-- change to "MR" if you want masks in MR space

# Resample BOTH images to this voxel size (mm) before registering.
COMMON_SPACING = (1.0, 1.0, 1.0)

# CT-space masks to carry into the fixed space.
MASKS = [
    "mandible.nii.gz",
    "parotid_gland_left.nii.gz",
    "parotid_gland_right.nii.gz",
    "spinal_cord.nii.gz",
]
# ===============================================================


# ---- STEP 1: load ----
print("Loading images...")
ct = ants.image_read(CT_PATH)
mr = ants.image_read(MR_PATH)
print(f"  CT  spacing={ct.spacing}  shape={ct.shape}")
print(f"  MR  spacing={mr.spacing}  shape={mr.shape}")

# ---- STEP 2: resample both to the same resolution ----
# interp_type=0 -> linear (correct for intensity images like CT / MR)
print(f"\nResampling both to {COMMON_SPACING} mm...")
ct_rs = ants.resample_image(ct, COMMON_SPACING, use_voxels=False, interp_type=0)
mr_rs = ants.resample_image(mr, COMMON_SPACING, use_voxels=False, interp_type=0)
print(f"  CT resampled shape={ct_rs.shape}")
print(f"  MR resampled shape={mr_rs.shape}")

# ---- assign fixed / moving based on your choice ----
if FIXED_MODALITY == "MR":
    fixed, moving = mr_rs, ct_rs
    print("\nFixed = MR, Moving = CT  (output will be in MR space)")
else:
    fixed, moving = ct_rs, mr_rs
    print("\nFixed = CT, Moving = MR  (output will be in CT space)")

# ---- STEP 3: register with SyN ----
# "SyNRA" = rigid -> affine -> SyN (deformable), the full gold-standard chain.
# metric="mattes" (mutual information) because CT and MR have different
# intensity scales (this is what makes it work across modalities).
print("\nRunning SyN registration (this takes a few minutes)...")
reg = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="SyNRA",
    aff_metric="mattes",
    syn_metric="mattes",
    verbose=False,
)
print("  Registration complete.")

# save the registered (warped) moving image
ants.image_write(reg["warpedmovout"],
                 os.path.join(OUT_DIR, "moving_registered_to_fixed.nii.gz"))
print("  Saved: moving_registered_to_fixed.nii.gz")

# ---- STEP 4: warp the CT masks into the fixed space ----
# This is only meaningful when CT is the MOVING image (FIXED_MODALITY = "MR"),
# since the masks live in CT space. If CT is fixed, the masks are already
# in the output space and don't need warping.
if FIXED_MODALITY == "MR":
    print("\nWarping CT masks into MR space (nearest-neighbour for labels)...")
    for fname in MASKS:
        mpath = os.path.join(SEG_DIR, fname)
        if not os.path.exists(mpath):
            print(f"  skip (not found): {fname}")
            continue
        mask = ants.image_read(mpath)
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=mask,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor",   # never blur a label mask
        )
        out = os.path.join(OUT_DIR, fname.replace(".nii.gz", "_registered.nii.gz"))
        ants.image_write(warped, out)
        print(f"  warped: {out}")
else:
    print("\nCT is the fixed image, so the CT masks are already in the output "
          "space. (Set FIXED_MODALITY='MR' if you want them moved into MR space.)")

print(f"\nDone. Everything saved in: {OUT_DIR}/")
