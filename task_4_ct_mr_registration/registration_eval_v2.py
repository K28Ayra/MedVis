"""
Registration Evaluation + Three-Plane Visualization — v2
=========================================================
Focus structures (available in CT TotalSegmentator output):
    sinus_frontal, sinus_maxillary, nasal_cavity (L+R),
    mandible, zygomatic_arch (L+R)

Note: caudate_nucleus, thalamus, tongue NOT available on CT
      (these are soft-tissue brain structures, require MRI-based segmentation)

Metrics:
    1. Mattes Mutual Information — before vs after registration
    2. Dice score — for Mandible (only structure with HaN-Seg GT)
    3. Visual three-plane overlays — all structures on registered MR

Run from inside case_01 folder:
    cd /scratch/ayra/hanseg/HaN-Seg/set_1/case_01/
    conda activate totalseg_env
    python3 registration_eval_v2.py
"""

import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

# ─── PATHS ────────────────────────────────────────────────────────────────
CASE_DIR  = "."
MR_PATH   = os.path.join(CASE_DIR, "case_01_IMG_MR_T1.nrrd")
CT_PATH   = os.path.join(CASE_DIR, "case_01_IMG_CT.nii.gz")
REG_DIR   = os.path.join(CASE_DIR, "registration_results")
SEG_DIR   = os.path.join(CASE_DIR, "case_01_seg")
OUT_DIR   = os.path.join(CASE_DIR, "registration_eval")
os.makedirs(OUT_DIR, exist_ok=True)

TRANSFORM_PATH  = os.path.join(REG_DIR, "ct_to_mr_rigid_transform.tfm")
CT_REG_PATH     = os.path.join(REG_DIR, "case_01_CT_registered_to_MR.nii.gz")

# ─── FOCUS STRUCTURES ─────────────────────────────────────────────────────
# (display_name, [pred_files], gt_file_or_None, color_hex)
STRUCTURES = [
    ("Frontal Sinus",
     ["sinus_frontal.nii.gz"],
     None, "#5bc8af"),

    ("Maxillary Sinus",
     ["sinus_maxillary.nii.gz"],
     None, "#e8c840"),

    ("Nasal Cavity",
     ["nasal_cavity_left.nii.gz", "nasal_cavity_right.nii.gz"],
     None, "#d080e0"),

    ("Mandible",
     ["mandible.nii.gz"],
     "case_01_OAR_Bone_Mandible.seg.nrrd", "#e05080"),

    ("Zygomatic Arch",
     ["zygomatic_arch_left.nii.gz", "zygomatic_arch_right.nii.gz"],
     None, "#50c840"),
]

# ─── HELPERS ──────────────────────────────────────────────────────────────
def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def norm(arr):
    mn, mx = float(arr.min()), float(arr.max())
    if mx == mn:
        return np.zeros_like(arr, dtype=float)
    return (arr.astype(float) - mn) / (mx - mn)

def make_overlay(base_slice, masks_colors, alpha=0.55):
    rgb = np.stack([norm(base_slice)]*3, axis=-1)
    for mask2d, color in masks_colors:
        if not mask2d.any():
            continue
        rgb[mask2d] = (1-alpha)*rgb[mask2d] + alpha*np.array(color)
    return rgb

def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return None
    return float(2*(pred & gt).sum() / denom)

# ─── STEP 1: LOAD MR ──────────────────────────────────────────────────────
print("="*60)
print("STEP 1: Loading MR image")
print("="*60)
mr_sitk = sitk.ReadImage(MR_PATH, sitk.sitkFloat32)
mr_arr  = sitk.GetArrayFromImage(mr_sitk)
print(f"  MR shape (z,y,x): {mr_arr.shape}")
print(f"  MR spacing: {[round(s,3) for s in mr_sitk.GetSpacing()]}")

# ─── STEP 2: MUTUAL INFORMATION ───────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Mutual Information (registration quality metric)")
print("="*60)

def compute_mi_value(fixed_sitk, moving_path, transform=None):
    """Compute MI between fixed image and moving image (optionally transformed)."""
    moving_sitk = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    if transform is not None:
        moving_sitk = sitk.Resample(
            moving_sitk, fixed_sitk, transform,
            sitk.sitkLinear, -1000.0, moving_sitk.GetPixelID()
        )
    else:
        moving_sitk = sitk.Resample(
            moving_sitk, fixed_sitk, sitk.Transform(),
            sitk.sitkLinear, -1000.0, moving_sitk.GetPixelID()
        )
    # Use registration method just to evaluate metric
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(sitk.TranslationTransform(3), inPlace=False)
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.0, numberOfIterations=1,
        convergenceMinimumValue=1.0, convergenceWindowSize=1
    )
    try:
        reg.Execute(
            sitk.Cast(fixed_sitk, sitk.sitkFloat32),
            sitk.Cast(moving_sitk, sitk.sitkFloat32)
        )
        return reg.GetMetricValue()
    except Exception as e:
        print(f"  MI computation error: {e}")
        return None

# Load saved transform
print("  Loading saved transform...")
transform = sitk.ReadTransform(TRANSFORM_PATH)

print("  Computing MI before registration (CT resampled to MR grid, no alignment)...")
mi_before = compute_mi_value(mr_sitk, CT_PATH, transform=None)

print("  Computing MI after registration (CT warped to MR space)...")
mi_after  = compute_mi_value(mr_sitk, CT_REG_PATH, transform=None)

print(f"\n  MI before registration: {mi_before:.6f}" if mi_before else "\n  MI before: could not compute")
print(f"  MI after  registration: {mi_after:.6f}"  if mi_after  else "  MI after:  could not compute")
if mi_before and mi_after:
    print(f"  Change: {mi_after - mi_before:+.6f}  ({'improved' if mi_after < mi_before else 'check alignment'})")
    print("  Note: More negative MI = better statistical alignment between CT and MR")

mi_df = pd.DataFrame([{
    "Metric": "Mattes Mutual Information",
    "Before registration": round(mi_before, 6) if mi_before else "N/A",
    "After registration":  round(mi_after,  6) if mi_after  else "N/A",
    "Interpretation": "More negative = better alignment. "
                      f"Change = {(mi_after-mi_before):+.6f}" if (mi_before and mi_after) else ""
}])
mi_csv = os.path.join(OUT_DIR, "MI_registration_metric.csv")
mi_df.to_csv(mi_csv, index=False)
print(f"  Saved: {mi_csv}")

# ─── STEP 3: APPLY TRANSFORM TO STRUCTURES ────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Applying transform to focus structures")
print("="*60)

registered_masks = {}

for name, pred_files, gt_file, color_hex in STRUCTURES:
    combined = None
    found    = False
    for fname in pred_files:
        fpath = os.path.join(SEG_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [{name}] WARNING: {fname} not found")
            continue
        m_sitk = sitk.ReadImage(fpath, sitk.sitkUInt8)
        m_reg  = sitk.Resample(
            m_sitk, mr_sitk, transform,
            sitk.sitkNearestNeighbor, 0, m_sitk.GetPixelID()
        )
        m_arr = sitk.GetArrayFromImage(m_reg) > 0
        combined = m_arr if combined is None else (combined | m_arr)
        found = True

    if not found or combined is None:
        print(f"  [{name}] SKIPPED — no files found")
        continue

    registered_masks[name] = (combined, color_hex)
    print(f"  [{name}] OK — {combined.sum()} voxels in MR space")

# ─── STEP 4: DICE SCORE (Mandible) ────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Dice score — Mandible vs HaN-Seg ground truth")
print("="*60)

dice_results = []
for name, pred_files, gt_file, color_hex in STRUCTURES:
    if gt_file is None:
        dice_results.append({
            "Structure": name, "DSC": "N/A",
            "Note": "No HaN-Seg ground truth — visual evaluation only"
        })
        continue

    gt_path = os.path.join(CASE_DIR, gt_file)
    if not os.path.exists(gt_path):
        print(f"  [{name}] GT file missing: {gt_path}")
        dice_results.append({"Structure": name, "DSC": "GT missing", "Note": gt_file})
        continue

    if name not in registered_masks:
        print(f"  [{name}] No registered mask")
        continue

    pred_arr, _ = registered_masks[name]

    # Resample GT into MR space
    gt_sitk     = sitk.ReadImage(gt_path, sitk.sitkUInt8)
    gt_resampled = sitk.Resample(
        gt_sitk, mr_sitk,
        sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, gt_sitk.GetPixelID()
    )
    gt_arr = sitk.GetArrayFromImage(gt_resampled) > 0

    print(f"  [{name}] pred voxels={pred_arr.sum()}  GT voxels={gt_arr.sum()}")
    print(f"  [{name}] pred shape={pred_arr.shape}  GT shape={gt_arr.shape}")

    if pred_arr.shape != gt_arr.shape:
        # crop to smaller
        s = tuple(min(a,b) for a,b in zip(pred_arr.shape, gt_arr.shape))
        pred_arr = pred_arr[:s[0],:s[1],:s[2]]
        gt_arr   = gt_arr[:s[0],:s[1],:s[2]]

    dsc = dice_score(pred_arr, gt_arr)
    dice_results.append({
        "Structure": name,
        "DSC": round(dsc, 4) if dsc is not None else "N/A",
        "Note": "TotalSegmentator (CT) vs HaN-Seg GT, after CT->MR registration"
    })
    print(f"  [{name}] DSC = {dsc:.4f}" if dsc else f"  [{name}] DSC = N/A")

df_dice = pd.DataFrame(dice_results)
dice_csv = os.path.join(OUT_DIR, "dice_results.csv")
df_dice.to_csv(dice_csv, index=False)
print(f"\n  Saved: {dice_csv}")
print(df_dice.to_string(index=False))

# ─── STEP 5: THREE-PLANE VISUALIZATION ────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Generating visualizations")
print("="*60)

plane_names = ['Axial', 'Coronal', 'Sagittal']

# Find best slices based on combined mask content
combined_all = np.zeros(mr_arr.shape, dtype=bool)
for name, (mask_arr, _) in registered_masks.items():
    if mask_arr.shape == mr_arr.shape:
        combined_all |= mask_arr

best_z = int(np.argmax(combined_all.sum(axis=(1,2))))
best_y = int(np.argmax(combined_all.sum(axis=(0,2))))
best_x = int(np.argmax(combined_all.sum(axis=(0,1))))
print(f"  Best slices: axial z={best_z}, coronal y={best_y}, sagittal x={best_x}")

mr_slices    = [mr_arr[best_z,:,:], mr_arr[:,best_y,:], mr_arr[:,:,best_x]]
slice_labels = [f'z={best_z}', f'y={best_y}', f'x={best_x}']

# --- Figure 1: Combined three-plane (MR only top, MR+masks bottom) ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#0d0d0d')
fig.suptitle('CT→MR Registration — case_01\nAll focus structures overlaid on MR',
             color='white', fontsize=14, fontweight='bold', y=0.98)

for col in range(3):
    # Row 0 — MR only
    axes[0,col].imshow(norm(mr_slices[col]), cmap='gray', origin='lower', aspect='auto')
    axes[0,col].set_title(f'{plane_names[col]}  ({slice_labels[col]})',
                          color='white', fontsize=11, pad=6)
    axes[0,col].axis('off'); axes[0,col].set_facecolor('black')
    if col == 0:
        axes[0,col].set_ylabel('MR only', color='white', fontsize=10,
                               rotation=90, labelpad=8)

    # Row 1 — MR + masks
    masks_for_col = []
    for name, (mask_arr, color_hex) in registered_masks.items():
        if mask_arr.shape != mr_arr.shape:
            continue
        slices = [mask_arr[best_z,:,:], mask_arr[:,best_y,:], mask_arr[:,:,best_x]]
        masks_for_col.append((slices[col], hex_to_rgb(color_hex)))

    axes[1,col].imshow(make_overlay(mr_slices[col], masks_for_col),
                       origin='lower', aspect='auto')
    axes[1,col].axis('off'); axes[1,col].set_facecolor('black')
    if col == 0:
        axes[1,col].set_ylabel('MR + registered masks', color='white',
                               fontsize=10, rotation=90, labelpad=8)

patches = [mpatches.Patch(color=hex_to_rgb(c), label=n)
           for n,(_, c) in registered_masks.items()]
axes[1,2].legend(handles=patches, loc='lower right', facecolor='#1a1a1a',
                 edgecolor='#444', labelcolor='white', fontsize=10, framealpha=0.9)

plt.tight_layout(rect=[0,0,1,0.96])
out1 = os.path.join(OUT_DIR, "threeplane_all_structures.png")
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"  Saved: {out1}")

# --- Figure 2: Per-structure three-plane ---
for name, (mask_arr, color_hex) in registered_masks.items():
    if mask_arr.shape != mr_arr.shape:
        continue
    # Best slice for THIS structure
    sz = int(np.argmax(mask_arr.sum(axis=(1,2))))
    sy = int(np.argmax(mask_arr.sum(axis=(0,2))))
    sx = int(np.argmax(mask_arr.sum(axis=(0,1))))

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d0d0d')
    fig2.suptitle(f'{name} — case_01 CT→MR Registration',
                  color='white', fontsize=13, fontweight='bold')

    for col, (mr_sl, mk_sl, plane, sl_lbl) in enumerate(zip(
        [mr_arr[sz,:,:], mr_arr[:,sy,:], mr_arr[:,:,sx]],
        [mask_arr[sz,:,:], mask_arr[:,sy,:], mask_arr[:,:,sx]],
        plane_names,
        [f'z={sz}', f'y={sy}', f'x={sx}']
    )):
        axes2[col].imshow(make_overlay(mr_sl, [(mk_sl, hex_to_rgb(color_hex))]),
                          origin='lower', aspect='auto')
        axes2[col].set_title(f'{plane}  ({sl_lbl})', color='white', fontsize=11)
        axes2[col].axis('off'); axes2[col].set_facecolor('black')

    axes2[2].legend(
        handles=[mpatches.Patch(color=hex_to_rgb(color_hex), label=name)],
        loc='lower right', facecolor='#1a1a1a', edgecolor='#444',
        labelcolor='white', fontsize=10
    )
    plt.tight_layout()
    safe = name.replace(' ', '_')
    out2 = os.path.join(OUT_DIR, f"{safe}_threeplane.png")
    plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"  Saved: {out2}")

# --- Figure 3: Dice bar chart ---
df_numeric = df_dice[pd.to_numeric(df_dice['DSC'], errors='coerce').notna()].copy()
df_numeric['DSC'] = pd.to_numeric(df_numeric['DSC'])
if not df_numeric.empty:
    fig3, ax = plt.subplots(figsize=(7, 3), facecolor='#0d0d0d')
    ax.set_facecolor('#111111')
    bar_colors = ['#5bc8af' if v>=0.75 else '#e8a040' if v>=0.5
                  else '#e05050' for v in df_numeric['DSC']]
    bars = ax.barh(df_numeric['Structure'], df_numeric['DSC'],
                   color=bar_colors, height=0.5)
    for bar, val in zip(bars, df_numeric['DSC']):
        ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', color='white', fontsize=11)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Dice Score (DSC)', color='white', fontsize=12)
    ax.set_title('Mandible — TotalSegmentator vs HaN-Seg GT\n(after CT→MR registration)',
                 color='white', fontsize=12)
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, "mandible_dice_chart.png")
    plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"  Saved: {out3}")

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL DONE")
print("="*60)
print(f"\nOutputs in: {OUT_DIR}/")
print("  MI_registration_metric.csv")
print("  dice_results.csv")
print("  threeplane_all_structures.png")
print("  <Structure>_threeplane.png  (one per structure)")
if not df_numeric.empty:
    print("  mandible_dice_chart.png")
print(f"\nMI after registration : {mi_after:.6f}" if mi_after else "")
print("\nDice results:")
print(df_dice.to_string(index=False))
