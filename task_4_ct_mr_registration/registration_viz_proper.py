"""
Proper Registration Visualization
===================================
For each structure shows 4 panels:
  Panel 1: CT scan + mask (where TotalSegmentator predicted it on CT)
  Panel 2: MR scan alone (what the MR looks like at same region)
  Panel 3: MR scan + registered mask (where it ended up after registration)
  Panel 4: CT registered (after reg) + mask (alignment check)

This makes it immediately clear what registration did.

Run from inside case_01 folder:
    cd /scratch/ayra/hanseg/HaN-Seg/set_1/case_01/
    conda activate totalseg_env
    python3 registration_viz_proper.py
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
REG_DIR   = os.path.join(CASE_DIR, "registration_results")
OUT_DIR   = os.path.join(CASE_DIR, "registration_viz_proper")
os.makedirs(OUT_DIR, exist_ok=True)

TRANSFORM_PATH = os.path.join(REG_DIR, "ct_to_mr_rigid_transform.tfm")
CT_REG_PATH    = os.path.join(REG_DIR, "case_01_CT_registered_to_MR.nii.gz")

# ─── STRUCTURES ───────────────────────────────────────────────────────────
STRUCTURES = [
    ("Mandible",       ["mandible.nii.gz"],                                      "#e05080"),
    ("Maxillary Sinus",["sinus_maxillary.nii.gz"],                               "#e8c840"),
    ("Nasal Cavity",   ["nasal_cavity_left.nii.gz","nasal_cavity_right.nii.gz"], "#d080e0"),
    ("Parotid L",      ["parotid_gland_left.nii.gz"],                            "#4a90d9"),
    ("Parotid R",      ["parotid_gland_right.nii.gz"],                           "#e87040"),
    ("Spinal Cord",    ["spinal_cord.nii.gz"],                                   "#5bc8af"),
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

def overlay(base, mask, color, alpha=0.6):
    rgb = np.stack([norm(base)]*3, axis=-1)
    if mask.any():
        rgb[mask] = (1-alpha)*rgb[mask] + alpha*np.array(color)
    return rgb

def best_slice(mask):
    """Find the axial slice with the most mask content."""
    scores = mask.sum(axis=(1,2))
    return int(np.argmax(scores)) if scores.max() > 0 else mask.shape[0]//2

# ─── LOAD IMAGES ──────────────────────────────────────────────────────────
print("Loading images...")
ct_sitk     = sitk.ReadImage(CT_PATH,    sitk.sitkFloat32)
mr_sitk     = sitk.ReadImage(MR_PATH,    sitk.sitkFloat32)
ct_reg_sitk = sitk.ReadImage(CT_REG_PATH,sitk.sitkFloat32)
transform   = sitk.ReadTransform(TRANSFORM_PATH)

ct_arr     = sitk.GetArrayFromImage(ct_sitk)      # z,y,x — original CT space
mr_arr     = sitk.GetArrayFromImage(mr_sitk)      # z,y,x — MR space
ct_reg_arr = sitk.GetArrayFromImage(ct_reg_sitk)  # z,y,x — CT warped to MR space

print(f"  CT  shape: {ct_arr.shape}")
print(f"  MR  shape: {mr_arr.shape}")
print(f"  CT_reg shape: {ct_reg_arr.shape}")

# ─── PER-STRUCTURE VISUALIZATION ──────────────────────────────────────────
for name, pred_files, color_hex in STRUCTURES:
    print(f"\nProcessing: {name}")
    color = hex_to_rgb(color_hex)

    # --- Load and merge prediction mask (in CT space) ---
    ct_mask = None
    for fname in pred_files:
        fpath = os.path.join(SEG_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found")
            continue
        m = sitk.GetArrayFromImage(sitk.ReadImage(fpath, sitk.sitkUInt8)) > 0
        ct_mask = m if ct_mask is None else (ct_mask | m)

    if ct_mask is None:
        print(f"  SKIPPED — no files found")
        continue

    # --- Register mask to MR space ---
    mr_mask = None
    for fname in pred_files:
        fpath = os.path.join(SEG_DIR, fname)
        if not os.path.exists(fpath):
            continue
        m_sitk = sitk.ReadImage(fpath, sitk.sitkUInt8)
        m_reg  = sitk.Resample(m_sitk, mr_sitk, transform,
                               sitk.sitkNearestNeighbor, 0, m_sitk.GetPixelID())
        m_arr  = sitk.GetArrayFromImage(m_reg) > 0
        mr_mask = m_arr if mr_mask is None else (mr_mask | m_arr)

    print(f"  CT mask voxels:  {ct_mask.sum()}")
    print(f"  MR mask voxels:  {mr_mask.sum()}")

    # --- Find best axial slice ---
    # For CT panels: use best slice in CT space
    # For MR panels: use best slice in MR space
    ct_z = best_slice(ct_mask)
    mr_z = best_slice(mr_mask) if mr_mask.sum() > 0 else mr_arr.shape[0]//2

    # Also find corresponding slice in registered CT
    # (CT_reg is in MR space, so use mr_z)
    ct_reg_z = min(mr_z, ct_reg_arr.shape[0]-1)

    print(f"  CT slice z={ct_z}, MR slice z={mr_z}")

    # --- Build the 4 panels ---
    # Panel 1: CT + mask (CT space)
    p1 = overlay(ct_arr[ct_z], ct_mask[ct_z], color)

    # Panel 2: MR alone (MR space, same slice as panel 3)
    p2 = norm(mr_arr[mr_z])

    # Panel 3: MR + registered mask (MR space)
    p3 = overlay(mr_arr[mr_z], mr_mask[mr_z], color) if mr_mask.sum() > 0 \
         else norm(mr_arr[mr_z])

    # Panel 4: Registered CT + registered mask (both in MR space)
    p4 = overlay(ct_reg_arr[ct_reg_z], mr_mask[ct_reg_z], color) \
         if mr_mask.sum() > 0 else norm(ct_reg_arr[ct_reg_z])

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), facecolor='#0d0d0d')
    fig.suptitle(
        f'{name} — CT→MR Registration  (case_01)',
        color='white', fontsize=14, fontweight='bold', y=1.02
    )

    panels = [
        (p1, f'CT + {name} mask\n(TotalSegmentator prediction)\nCT slice z={ct_z}', True),
        (p2, f'MR scan (fixed image)\nno mask\nMR slice z={mr_z}', False),
        (p3, f'MR + registered mask\n(after CT→MR registration)\nMR slice z={mr_z}', True),
        (p4, f'Registered CT + mask\n(alignment check, both in MR space)\nz={ct_reg_z}', True),
    ]

    for ax, (img, title, is_rgb) in zip(axes, panels):
        if is_rgb:
            ax.imshow(img, origin='lower', aspect='auto')
        else:
            ax.imshow(img, cmap='gray', origin='lower', aspect='auto')
        ax.set_title(title, color='white', fontsize=9, pad=6)
        ax.axis('off')
        ax.set_facecolor('black')

    # Add status note
    if mr_mask.sum() == 0:
        status = "⚠ Outside MR field of view"
        status_color = '#e87040'
    elif mr_mask.sum() < 100:
        status = "⚠ Very few voxels registered — partially outside MR FOV"
        status_color = '#e87040'
    else:
        status = "✓ Successfully registered to MR space"
        status_color = '#5bc8af'

    fig.text(0.5, -0.02, status, ha='center', color=status_color,
             fontsize=11, fontweight='bold')

    patch = mpatches.Patch(color=color, label=name)
    axes[3].legend(handles=[patch], loc='lower right',
                   facecolor='#1a1a1a', edgecolor='#444',
                   labelcolor='white', fontsize=10)

    plt.tight_layout()
    safe = name.replace(' ', '_')
    out = os.path.join(OUT_DIR, f"{safe}_registration.png")
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d', pad_inches=0.3)
    plt.close()
    print(f"  Saved: {out}")

# ─── COMBINED SUMMARY FIGURE ──────────────────────────────────────────────
# Show MR + all successfully registered masks together
print("\nGenerating combined summary figure...")

valid_masks = []
valid_colors = []
valid_names  = []

for name, pred_files, color_hex in STRUCTURES:
    color = hex_to_rgb(color_hex)
    mr_mask = None
    for fname in pred_files:
        fpath = os.path.join(SEG_DIR, fname)
        if not os.path.exists(fpath):
            continue
        m_sitk = sitk.ReadImage(fpath, sitk.sitkUInt8)
        m_reg  = sitk.Resample(m_sitk, mr_sitk, transform,
                               sitk.sitkNearestNeighbor, 0, m_sitk.GetPixelID())
        m_arr  = sitk.GetArrayFromImage(m_reg) > 0
        mr_mask = m_arr if mr_mask is None else (mr_mask | m_arr)

    if mr_mask is not None and mr_mask.sum() > 100:
        valid_masks.append(mr_mask)
        valid_colors.append(color)
        valid_names.append(name)

if valid_masks:
    combined = np.zeros(mr_arr.shape, dtype=bool)
    for m in valid_masks:
        if m.shape == mr_arr.shape:
            combined |= m

    best_z_ax  = int(np.argmax(combined.sum(axis=(1,2))))
    best_y_cor = int(np.argmax(combined.sum(axis=(0,2))))
    best_x_sag = int(np.argmax(combined.sum(axis=(0,1))))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#0d0d0d')
    fig.suptitle('CT→MR Registration — case_01\nAll structures in MR space',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    planes     = ['Axial', 'Coronal', 'Sagittal']
    mr_slices  = [mr_arr[best_z_ax,:,:],
                  mr_arr[:,best_y_cor,:],
                  mr_arr[:,:,best_x_sag]]
    sl_labels  = [f'z={best_z_ax}', f'y={best_y_cor}', f'x={best_x_sag}']

    for col in range(3):
        # Row 0 — MR only
        axes[0,col].imshow(norm(mr_slices[col]), cmap='gray',
                           origin='lower', aspect='auto')
        axes[0,col].set_title(f'{planes[col]}  ({sl_labels[col]})',
                              color='white', fontsize=11, pad=6)
        axes[0,col].axis('off')
        axes[0,col].set_facecolor('black')
        if col == 0:
            axes[0,col].set_ylabel('MR only', color='white',
                                   fontsize=11, rotation=90, labelpad=8)

        # Row 1 — MR + all masks
        rgb = np.stack([norm(mr_slices[col])]*3, axis=-1)
        for mask, color in zip(valid_masks, valid_colors):
            if mask.shape != mr_arr.shape:
                continue
            slices = [mask[best_z_ax,:,:],
                      mask[:,best_y_cor,:],
                      mask[:,:,best_x_sag]]
            m2d = slices[col]
            if m2d.any():
                rgb[m2d] = 0.4*rgb[m2d] + 0.6*np.array(color)

        axes[1,col].imshow(rgb, origin='lower', aspect='auto')
        axes[1,col].axis('off')
        axes[1,col].set_facecolor('black')
        if col == 0:
            axes[1,col].set_ylabel('MR + registered masks',
                                   color='white', fontsize=11,
                                   rotation=90, labelpad=8)

    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(valid_colors, valid_names)]
    axes[1,2].legend(handles=patches, loc='lower right',
                     facecolor='#1a1a1a', edgecolor='#444',
                     labelcolor='white', fontsize=10, framealpha=0.9)

    plt.tight_layout(rect=[0,0,1,0.96])
    out_sum = os.path.join(OUT_DIR, "summary_all_structures.png")
    plt.savefig(out_sum, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"Saved: {out_sum}")

# ─── DONE ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("ALL DONE")
print("="*55)
print(f"\nOutputs saved to: {OUT_DIR}/")
print("  <Structure>_registration.png  — 4-panel per structure")
print("  summary_all_structures.png    — combined overview")
print("\nEach 4-panel figure shows:")
print("  Panel 1: CT + mask (where TotalSegmentator found it)")
print("  Panel 2: MR alone (what MR looks like at that region)")
print("  Panel 3: MR + mask (where it landed after registration)")
print("  Panel 4: Registered CT + mask (alignment check)")
