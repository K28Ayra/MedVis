"""
Registration RESULTS + QC  —  CT-fixed SyN registration (case_01)
=================================================================
Shows how well the warped MR lines up with the fixed CT, and whether
SyN improved the alignment (before vs after).

Produces (in registration_syn/results/):
  1. checkerboard_before_after.png  -> the key alignment figure
  2. fusion_overlay.png             -> CT (gray) + MR (hot) blended
  3. edge_overlay.png               -> CT bone edges drawn on the MR
  4. mutual_information.png + .csv  -> quantitative: MI before vs after

Run on the SERVER (all data lives there):
    cd /scratch/ayra/hanseg/HaN-Seg/set_1/case_01/
    conda activate totalseg_env
    python3 registration_results.py
"""

import SimpleITK as sitk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================ CONFIG ============================
CASE_DIR   = "."
CT_PATH    = os.path.join(CASE_DIR, "case_01_IMG_CT.nii.gz")
MR_PATH    = os.path.join(CASE_DIR, "case_01_IMG_MR_T1.nrrd")
WARPED_MR  = os.path.join(CASE_DIR, "registration_syn",
                          "moving_registered_to_fixed.nii.gz")
OUT_DIR    = os.path.join(CASE_DIR, "registration_syn", "results")
os.makedirs(OUT_DIR, exist_ok=True)

BG = "#0d0d0d"     # figure background (matches your earlier style)
# ===============================================================


# ---------- helpers ----------
def resample_like(img, ref, interp=sitk.sitkLinear, default=0.0):
    """Put `img` onto `ref`'s grid using identity transform (world coords)."""
    return sitk.Resample(img, ref, sitk.Transform(), interp,
                         default, img.GetPixelID())

def arr(img):
    return sitk.GetArrayFromImage(img)   # z, y, x

def norm(a, lo=1, hi=99):
    a = a.astype(np.float32)
    pos = a[a > a.min()]
    if pos.size == 0:
        return np.zeros_like(a)
    vlo, vhi = np.percentile(pos, lo), np.percentile(pos, hi)
    if vhi <= vlo:
        vhi = vlo + 1e-6
    return np.clip((a - vlo) / (vhi - vlo), 0, 1)

def checkerboard(a, b, tiles=10):
    """Blend two normalized 2D slices in a checker pattern."""
    h, w = a.shape
    ty, tx = max(1, h // tiles), max(1, w // tiles)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy // ty) + (xx // tx)) % 2 == 0
    out = np.where(mask, a, b)
    return out

def mutual_information(a, b, bins=64, sample=400000):
    """MI between two same-shape volumes (subsampled for speed)."""
    a = a.ravel(); b = b.ravel()
    keep = (a > a.min())            # ignore pure background/air
    a, b = a[keep], b[keep]
    if a.size > sample:
        idx = np.random.default_rng(0).choice(a.size, sample, replace=False)
        a, b = a[idx], b[idx]
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist / hist.sum()
    px = pxy.sum(1); py = pxy.sum(0)
    nz = pxy > 0
    return float((pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])).sum())

def three_planes(vol):
    """Return (axial, coronal, sagittal) mid-content slices as indices."""
    m = vol > np.percentile(vol, 60)
    z = int(np.argmax(m.sum((1, 2)))); y = int(np.argmax(m.sum((0, 2)))); x = int(np.argmax(m.sum((0, 1))))
    return z, y, x

def show(ax, img, cmap='gray'):
    ax.imshow(img, cmap=cmap, origin='lower', aspect='equal')
    ax.axis('off'); ax.set_facecolor('black')


# ---------- load ----------
print("Loading CT, original MR, warped MR...")
ct_raw   = sitk.ReadImage(CT_PATH,   sitk.sitkFloat32)
mr_raw    = sitk.ReadImage(MR_PATH,   sitk.sitkFloat32)
warped   = sitk.ReadImage(WARPED_MR, sitk.sitkFloat32)

# put CT and the ORIGINAL (unregistered) MR on the warped MR's grid,
# so all three share one identical grid for fair comparison
ct     = resample_like(ct_raw, warped, sitk.sitkLinear, -1000.0)
mr_before = resample_like(mr_raw, warped, sitk.sitkLinear, 0.0)

CT   = arr(ct)
BEF  = arr(mr_before)     # MR before registration, in CT space
AFT  = arr(warped)        # MR after SyN, in CT space
print(f"  grid: {CT.shape}  (all three aligned to this)")

z, y, x = three_planes(CT)

# extract only the slices we need, THEN normalize (avoids full-volume copies)
def raw_slc(v, i):
    return [v[z, :, :], v[:, y, :], v[:, :, x]][i]

CT_S  = [norm(raw_slc(CT,  i)) for i in range(3)]
BEF_S = [norm(raw_slc(BEF, i)) for i in range(3)]
AFT_S = [norm(raw_slc(AFT, i)) for i in range(3)]

def slcCT(i):  return CT_S[i]
def slcBEF(i): return BEF_S[i]
def slcAFT(i): return AFT_S[i]


# ---------- FIG 1: checkerboard before vs after ----------
print("Figure 1: checkerboard before/after...")
fig, axs = plt.subplots(2, 3, figsize=(16, 11), facecolor=BG)
fig.suptitle("CT + MR alignment — checkerboard (smooth edges across squares = good)",
             color='white', fontsize=14, fontweight='bold', y=0.98)
titles = ["Axial", "Coronal", "Sagittal"]
for c in range(3):
    show(axs[0, c], checkerboard(slcCT(c), slcBEF(c)))
    axs[0, c].set_title(f"{titles[c]} — BEFORE (no registration)", color='#e8a040', fontsize=11)
    show(axs[1, c], checkerboard(slcCT(c), slcAFT(c)))
    axs[1, c].set_title(f"{titles[c]} — AFTER (SyN)", color='#5bc8af', fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT_DIR, "checkerboard_before_after.png"),
            dpi=140, facecolor=BG, bbox_inches='tight'); plt.close()

# ---------- FIG 2: fusion overlay (after) ----------
print("Figure 2: fusion overlay...")
fig, axs = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
fig.suptitle("CT (gray) + registered MR (hot) — fusion overlay",
             color='white', fontsize=13, fontweight='bold')
for c in range(3):
    axs[c].imshow(slcCT(c), cmap='gray', origin='lower', aspect='equal')
    axs[c].imshow(slcAFT(c), cmap='hot', alpha=0.45, origin='lower', aspect='equal')
    axs[c].set_title(titles[c], color='white', fontsize=11)
    axs[c].axis('off'); axs[c].set_facecolor('black')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fusion_overlay.png"),
            dpi=140, facecolor=BG, bbox_inches='tight'); plt.close()

# ---------- FIG 3: CT edges on MR ----------
print("Figure 3: edge overlay...")
from scipy import ndimage
def edges(a2d):
    gx = ndimage.sobel(a2d, axis=0); gy = ndimage.sobel(a2d, axis=1)
    g = np.hypot(gx, gy)
    return g > np.percentile(g, 96)
fig, axs = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
fig.suptitle("CT bone edges (cyan) over registered MR — should trace MR structures",
             color='white', fontsize=13, fontweight='bold')
for c in range(3):
    base = np.stack([slcAFT(c)]*3, -1)
    e = edges(slcCT(c))
    base[e] = [0.2, 0.9, 0.9]
    axs[c].imshow(base, origin='lower', aspect='equal')
    axs[c].set_title(titles[c], color='white', fontsize=11)
    axs[c].axis('off'); axs[c].set_facecolor('black')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "edge_overlay.png"),
            dpi=140, facecolor=BG, bbox_inches='tight'); plt.close()

# ---------- FIG 4: mutual information before vs after ----------
print("Figure 4: mutual information (this takes a moment)...")
mi_before = mutual_information(CT, BEF)
mi_after  = mutual_information(CT, AFT)
with open(os.path.join(OUT_DIR, "mutual_information.csv"), "w") as f:
    f.write("stage,mutual_information\n")
    f.write(f"before,{mi_before:.4f}\n")
    f.write(f"after,{mi_after:.4f}\n")

fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=BG)
ax.set_facecolor('#111')
bars = ax.bar(["Before\n(no reg)", "After\n(SyN)"], [mi_before, mi_after],
              color=['#e8a040', '#5bc8af'], width=0.6)
for b, v in zip(bars, [mi_before, mi_after]):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha='center', va='bottom',
            color='white', fontsize=12, fontweight='bold')
ax.set_ylabel("Mutual Information  (higher = better aligned)", color='white')
ax.set_title("CT ↔ MR alignment quality", color='white', fontsize=13)
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#333')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mutual_information.png"),
            dpi=140, facecolor=BG, bbox_inches='tight'); plt.close()

print("\n" + "="*55)
print("RESULTS GENERATED")
print("="*55)
print(f"Saved to: {OUT_DIR}/")
print(f"  checkerboard_before_after.png")
print(f"  fusion_overlay.png")
print(f"  edge_overlay.png")
print(f"  mutual_information.png / .csv")
print(f"\n  MI before = {mi_before:.4f}")
print(f"  MI after  = {mi_after:.4f}")
print(f"  change    = {mi_after - mi_before:+.4f}")
