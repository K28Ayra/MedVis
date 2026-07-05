"""
Per-structure alignment QC  —  CT-fixed registration (case_01)
==============================================================
For each organ: draws the mask OUTLINE on the CT (where it was defined)
and on the WARPED MR (registered into CT space). If the outline traces
the same anatomy in both, that structure is well aligned.

Produces (in registration_syn/results/structures/):
  overview_all_structures.png     -> all organs colored on the MR, 3 planes
  <Structure>_alignment.png       -> one per organ: CT vs MR, 3 planes, with outline

Run on the SERVER:
    cd /scratch/ayra/hanseg/HaN-Seg/set_1/case_01/
    conda activate totalseg_env
    python3 structure_alignment.py
"""

import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, gc

# ============================ CONFIG ============================
CASE_DIR  = "."
CT_PATH   = os.path.join(CASE_DIR, "case_01_IMG_CT.nii.gz")
WARPED_MR = os.path.join(CASE_DIR, "registration_syn",
                         "moving_registered_to_fixed.nii.gz")
SEG_DIR   = os.path.join(CASE_DIR, "case_01_seg")
OUT_DIR   = os.path.join(CASE_DIR, "registration_syn", "results", "structures")
os.makedirs(OUT_DIR, exist_ok=True)

# (display name, filename in seg dir, outline color)
STRUCTURES = [
    ("Mandible",     "mandible.nii.gz",             "#e8c840"),
    ("Parotid L",    "parotid_gland_left.nii.gz",   "#4a90d9"),
    ("Parotid R",    "parotid_gland_right.nii.gz",  "#e87040"),
    ("Spinal Cord",  "spinal_cord.nii.gz",          "#5bc8af"),
]

BG = "#0d0d0d"
# ===============================================================


def resample_like(img, ref, interp, default):
    return sitk.Resample(img, ref, sitk.Transform(), interp, default, img.GetPixelID())

def norm(a, lo=1, hi=99):
    a = a.astype(np.float32)
    pos = a[a > a.min()]
    if pos.size == 0:
        return np.zeros_like(a)
    vlo, vhi = np.percentile(pos, lo), np.percentile(pos, hi)
    if vhi <= vlo: vhi = vlo + 1e-6
    return np.clip((a - vlo) / (vhi - vlo), 0, 1)

def hex_rgb(h):
    h = h.lstrip('#'); return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))


# ---- load fixed CT + warped MR onto one shared grid ----
print("Loading CT and warped MR...")
warped = sitk.ReadImage(WARPED_MR, sitk.sitkFloat32)
ct     = resample_like(sitk.ReadImage(CT_PATH, sitk.sitkFloat32),
                       warped, sitk.sitkLinear, -1000.0)
CT  = sitk.GetArrayFromImage(ct)       # z,y,x
MR  = sitk.GetArrayFromImage(warped)
print(f"  shared grid: {CT.shape}")

planes = ["Axial", "Coronal", "Sagittal"]

def slices_at(vol, z, y, x):
    return [vol[z, :, :], vol[:, y, :], vol[:, :, x]]


# ---- load + resample each mask onto the shared grid ----
loaded = []   # (name, mask_bool_array, color_hex)
for name, fname, color in STRUCTURES:
    p = os.path.join(SEG_DIR, fname)
    if not os.path.exists(p):
        print(f"  skip (missing): {fname}"); continue
    m = resample_like(sitk.ReadImage(p, sitk.sitkUInt8),
                      warped, sitk.sitkNearestNeighbor, 0)
    ma = sitk.GetArrayFromImage(m) > 0
    if ma.sum() == 0:
        print(f"  skip (empty after resample): {name}"); continue
    loaded.append((name, ma, color))
    print(f"  {name}: {ma.sum()} voxels")
    del m; gc.collect()

if not loaded:
    raise RuntimeError("No masks loaded — check SEG_DIR and filenames.")


# ---- OVERVIEW: all structures colored on the MR ----
print("Overview figure...")
allmask = np.zeros_like(CT, dtype=bool)
for _, ma, _ in loaded: allmask |= ma
z = int(np.argmax(allmask.sum((1,2)))); y = int(np.argmax(allmask.sum((0,2)))); x = int(np.argmax(allmask.sum((0,1))))
mr_sl = slices_at(MR, z, y, x)

fig, axs = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
fig.suptitle("All structures on registered MR (outlines)", color='white',
             fontsize=13, fontweight='bold')
for c in range(3):
    axs[c].imshow(norm(mr_sl[c]), cmap='gray', origin='lower', aspect='equal')
    for name, ma, color in loaded:
        s = slices_at(ma, z, y, x)[c].astype(float)
        if s.any():
            axs[c].contour(s, levels=[0.5], colors=[hex_rgb(color)], linewidths=1.2)
    axs[c].set_title(planes[c], color='white', fontsize=11)
    axs[c].axis('off'); axs[c].set_facecolor('black')
axs[2].legend(handles=[mpatches.Patch(color=hex_rgb(c), label=n)
                       for n, _, c in loaded],
              loc='lower right', facecolor='#1a1a1a', edgecolor='#444',
              labelcolor='white', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "overview_all_structures.png"),
            dpi=140, facecolor=BG, bbox_inches='tight'); plt.close()


# ---- PER STRUCTURE: CT row vs MR row, outline on both ----
for name, ma, color in loaded:
    zz = int(np.argmax(ma.sum((1,2)))); yy = int(np.argmax(ma.sum((0,2)))); xx = int(np.argmax(ma.sum((0,1))))
    ct_sl = slices_at(CT, zz, yy, xx)
    mr_sl = slices_at(MR, zz, yy, xx)
    m_sl  = slices_at(ma, zz, yy, xx)
    col   = hex_rgb(color)

    fig, axs = plt.subplots(2, 3, figsize=(16, 11), facecolor=BG)
    fig.suptitle(f"{name} — outline on CT (top) vs registered MR (bottom)\n"
                 f"aligned if the outline traces the same shape in both rows",
                 color='white', fontsize=13, fontweight='bold', y=0.98)
    for c in range(3):
        axs[0, c].imshow(norm(ct_sl[c]), cmap='gray', origin='lower', aspect='equal')
        axs[1, c].imshow(norm(mr_sl[c]), cmap='gray', origin='lower', aspect='equal')
        s = m_sl[c].astype(float)
        if s.any():
            axs[0, c].contour(s, levels=[0.5], colors=[col], linewidths=1.5)
            axs[1, c].contour(s, levels=[0.5], colors=[col], linewidths=1.5)
        axs[0, c].set_title(f"CT — {planes[c]}", color='#bbbbbb', fontsize=10)
        axs[1, c].set_title(f"MR — {planes[c]}", color=col, fontsize=10)
        for r in (0, 1):
            axs[r, c].axis('off'); axs[r, c].set_facecolor('black')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe = name.replace(" ", "_")
    plt.savefig(os.path.join(OUT_DIR, f"{safe}_alignment.png"),
                dpi=140, facecolor=BG, bbox_inches='tight'); plt.close()
    print(f"  saved: {safe}_alignment.png")

print(f"\nDone. Figures in: {OUT_DIR}/")
