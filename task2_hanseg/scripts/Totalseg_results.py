"""
TotalSegmentator (CT) vs HaN-Seg ground truth — Dice scoring + visualization
Run this from inside the case_01 folder on the remote server:
    cd /scratch/ayra/hanseg/HaN-Seg/set_1/case_01/
    python3 totalseg_results.py

Needs: pip install matplotlib pandas   (SimpleITK + numpy already installed)
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
CASE_DIR = "."
PRED_DIR = os.path.join(CASE_DIR, "case_01_seg")
CT_PATH  = os.path.join(CASE_DIR, "case_01_IMG_CT.nii.gz")
OUT_DIR  = os.path.join(CASE_DIR, "totalseg_results")
os.makedirs(OUT_DIR, exist_ok=True)

def gt_path(name):
    return os.path.join(CASE_DIR, f"case_01_OAR_{name}.seg.nrrd")

def pred_path(name):
    return os.path.join(PRED_DIR, f"{name}.nii.gz")

# ─── MATCHING TABLE: TotalSegmentator output(s)  ->  HaN-Seg ground truth ──
# Most are 1-to-1. Carotid is a union of TotalSegmentator's two separate
# vessel classes, since HaN-Seg annotates the carotid as one structure.
# Esophagus is flagged approximate: TotalSegmentator predicts the whole
# esophagus, HaN-Seg ground truth (Esophagus_S) is only the cervical part.
MATCHES = [
    {"display": "Parotid L",        "preds": ["parotid_gland_left"],                                   "gt": "Parotid_L",      "note": ""},
    {"display": "Parotid R",        "preds": ["parotid_gland_right"],                                  "gt": "Parotid_R",      "note": ""},
    {"display": "Submandibular L",  "preds": ["submandibular_gland_left"],                              "gt": "Glnd_Submand_L", "note": ""},
    {"display": "Submandibular R",  "preds": ["submandibular_gland_right"],                             "gt": "Glnd_Submand_R", "note": ""},
    {"display": "Optic Nerve L",    "preds": ["optic_nerve_left"],                                      "gt": "OpticNrv_L",     "note": ""},
    {"display": "Optic Nerve R",    "preds": ["optic_nerve_right"],                                     "gt": "OpticNrv_R",     "note": ""},
    {"display": "Spinal Cord",      "preds": ["spinal_cord"],                                           "gt": "SpinalCord",     "note": ""},
    {"display": "Thyroid",          "preds": ["thyroid_gland"],                                         "gt": "Glnd_Thyroid",   "note": ""},
    {"display": "Carotid L",        "preds": ["common_carotid_artery_left", "internal_carotid_artery_left"],   "gt": "A_Carotid_L", "note": "approx: union of 2 TS classes"},
    {"display": "Carotid R",        "preds": ["common_carotid_artery_right", "internal_carotid_artery_right"], "gt": "A_Carotid_R", "note": "approx: union of 2 TS classes"},
    {"display": "Esophagus (S)",    "preds": ["esophagus"],                                             "gt": "Esophagus_S",    "note": "approx: TS covers full esophagus, GT is cervical only"},
]

# Organs in HaN-Seg with NO TotalSegmentator equivalent at all (for the coverage chart)
NOT_COVERED = [
    "Brainstem", "OpticChiasm", "Cochlea_L", "Cochlea_R", "Pituitary",
    "Bone_Mandible", "Arytenoid", "Cricopharyngeus", "Glottis", "Larynx_SG",
    "Lips", "Cavity_Oral", "Glnd_Lacrimal_L", "Glnd_Lacrimal_R",
    "Eye_AL", "Eye_AR", "Eye_PL", "Eye_PR",
]

COLORS = ['#4a90d9','#5bc8af','#e87040','#e8c840','#e05080','#d080e0',
          '#50c840','#e8a040','#c05050','#8050d0','#608030']

# ─── HELPERS ──────────────────────────────────────────────────────────────
def load_array(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)  # z, y, x

def dice(pred_mask, gt_mask):
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return None
    return float(2 * (pred_mask & gt_mask).sum() / denom)

def union_pred_mask(names, shape):
    mask = np.zeros(shape, dtype=bool)
    found_any = False
    for n in names:
        p = pred_path(n)
        if os.path.exists(p):
            mask |= (load_array(p) > 0)
            found_any = True
        else:
            print(f"  WARNING: prediction file missing: {p}")
    return mask, found_any

# ─── STEP 1: LOAD CT + COMPUTE DICE FOR EACH MATCHED STRUCTURE ─────────────
print("Loading CT...")
ct = load_array(CT_PATH)
print(f"  CT shape: {ct.shape}")

results = []
gt_masks_for_viz = {}   # display name -> mask
pred_masks_for_viz = {}

for m in MATCHES:
    gtp = gt_path(m["gt"])
    if not os.path.exists(gtp):
        print(f"  WARNING: ground truth missing: {gtp} — skipping {m['display']}")
        continue

    gt_mask = load_array(gtp) > 0
    pred_mask, found = union_pred_mask(m["preds"], gt_mask.shape)

    if not found:
        print(f"  WARNING: no prediction found for {m['display']} — skipping")
        continue
    if gt_mask.shape != pred_mask.shape:
        print(f"  WARNING: shape mismatch for {m['display']}: GT {gt_mask.shape} vs pred {pred_mask.shape} — skipping")
        continue

    d = dice(pred_mask, gt_mask)
    results.append({"structure": m["display"], "DSC": round(d, 4) if d is not None else None,
                     "note": m["note"]})
    print(f"  {m['display']:18s} DSC = {d:.4f}" if d is not None else f"  {m['display']:18s} DSC = N/A (empty masks)")

    gt_masks_for_viz[m["display"]] = gt_mask
    pred_masks_for_viz[m["display"]] = pred_mask

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUT_DIR, "totalseg_dice_results.csv"), index=False)
mean_dsc = df["DSC"].mean()
print(f"\nMean DSC across {len(df)} matched structures: {mean_dsc:.4f}")
print(f"Saved: {OUT_DIR}/totalseg_dice_results.csv")

# ─── STEP 2: DICE BAR CHART ─────────────────────────────────────────────────
df_sorted = df.dropna(subset=["DSC"]).sort_values("DSC", ascending=True)
bar_colors = ['#e05050' if v < 0.5 else '#e8a040' if v < 0.75 else '#5bc8af'
              for v in df_sorted["DSC"]]

fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d0d0d')
ax.set_facecolor('#111111')
bars = ax.barh(df_sorted["structure"], df_sorted["DSC"], color=bar_colors, height=0.6)
for bar, val in zip(bars, df_sorted["DSC"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', fontsize=9, color='white')
ax.axvline(x=mean_dsc, color='white', lw=1.5, label=f'Mean = {mean_dsc:.3f}')
ax.set_xlim(0, 1.1)
ax.set_xlabel('Dice Score (DSC)', color='white', fontsize=12)
ax.set_title(f'TotalSegmentator vs HaN-Seg GT — case_01  |  Mean DSC = {mean_dsc:.3f}\n'
             f'({len(df_sorted)} of 30 HaN-Seg OARs have a TotalSegmentator equivalent)',
             color='white', fontsize=12, pad=10)
ax.tick_params(colors='white', labelsize=10)
for sp in ax.spines.values(): sp.set_edgecolor('#333')
patches = [
    mpatches.Patch(color='#5bc8af', label='≥ 0.75 (good)'),
    mpatches.Patch(color='#e8a040', label='0.50–0.75 (acceptable)'),
    mpatches.Patch(color='#e05050', label='< 0.50 (poor)'),
    plt.Line2D([0],[0], color='white', lw=1.5, label=f'Mean DSC = {mean_dsc:.3f}')
]
ax.legend(handles=patches, facecolor='#1a1a1a', edgecolor='#333',
          labelcolor='white', fontsize=9, loc='lower right')
plt.tight_layout()
out = os.path.join(OUT_DIR, "totalseg_dice_chart.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"Saved: {out}")

# ─── STEP 3: COVERAGE CHART (all 30 HaN-Seg OARs) ──────────────────────────
covered = [m["display"] for m in MATCHES if m["display"] in gt_masks_for_viz]
n_covered = len(covered)
n_not_covered = len(NOT_COVERED)

fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0d0d0d')
ax.set_facecolor('#111111')
labels = ['Covered by\nTotalSegmentator', 'No TotalSegmentator\nequivalent']
counts = [n_covered, n_not_covered]
colors_cov = ['#5bc8af', '#555555']
bars = ax.bar(labels, counts, color=colors_cov, width=0.5)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, c+0.3, str(c), ha='center',
            color='white', fontsize=13, fontweight='bold')
ax.set_ylim(0, 30)
ax.set_ylabel('Number of HaN-Seg OARs (out of 30)', color='white', fontsize=11)
ax.set_title('TotalSegmentator (CT) Coverage of HaN-Seg OARs', color='white', fontsize=13, pad=10)
ax.tick_params(colors='white', labelsize=10)
for sp in ax.spines.values(): sp.set_edgecolor('#333')
plt.tight_layout()
out = os.path.join(OUT_DIR, "totalseg_coverage_chart.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"Saved: {out}")

# ─── STEP 4: OVERLAY VISUALIZATION AT BEST SLICE ───────────────────────────
# pick the axial slice with the most combined matched-structure area in GT
slice_scores = np.zeros(ct.shape[0])
for mask in gt_masks_for_viz.values():
    slice_scores += mask.sum(axis=(1,2))
best_z = int(np.argmax(slice_scores))
print(f"\nUsing axial slice z={best_z} for overlay visualization (largest combined OAR area)")

def make_overlay(ct_slice, masks_dict, alpha=0.55):
    n = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
    rgb = np.stack([n]*3, axis=-1)
    for i, (name, mask) in enumerate(masks_dict.items()):
        col = COLORS[i % len(COLORS)]
        r,g,b = int(col[1:3],16)/255, int(col[3:5],16)/255, int(col[5:7],16)/255
        m = mask[best_z]
        if not m.any(): continue
        rgb[m] = (1-alpha)*rgb[m] + alpha*np.array([r,g,b])
    return rgb

ct_slice = ct[best_z]
fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor='#0d0d0d')
fig.suptitle(f'TotalSegmentator vs HaN-Seg GT — case_01, axial z={best_z}', color='white', fontsize=13)

n = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
axes[0].imshow(n, cmap='gray', origin='upper')
axes[0].set_title('CT', color='white', fontsize=11)
axes[1].imshow(make_overlay(ct_slice, gt_masks_for_viz), origin='upper')
axes[1].set_title('HaN-Seg Ground Truth', color='white', fontsize=11)
axes[2].imshow(make_overlay(ct_slice, pred_masks_for_viz), origin='upper')
axes[2].set_title('TotalSegmentator Prediction', color='white', fontsize=11)

patches = [mpatches.Patch(color=COLORS[i % len(COLORS)], label=name)
           for i, name in enumerate(gt_masks_for_viz.keys())]
axes[2].legend(handles=patches, facecolor='#1a1a1a', edgecolor='#333',
               labelcolor='white', fontsize=7, loc='upper right', ncol=1)

for ax in axes:
    ax.axis('off')
    ax.set_facecolor('black')
plt.tight_layout()
out = os.path.join(OUT_DIR, "totalseg_overlay_comparison.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"Saved: {out}")

# ─── FINAL SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TOTALSEGMENTATOR (CT) RESULTS SUMMARY — case_01")
print("="*60)
print(f"HaN-Seg OARs with a TotalSegmentator match: {n_covered} / 30")
print(f"HaN-Seg OARs with NO TotalSegmentator match: {n_not_covered} / 30")
print(f"Mean DSC across matched structures:          {mean_dsc:.4f}")
if not df_sorted.empty:
    best = df_sorted.iloc[-1]
    worst = df_sorted.iloc[0]
    print(f"Best structure:   {best['structure']} ({best['DSC']:.4f})")
    print(f"Worst structure:  {worst['structure']} ({worst['DSC']:.4f})")
print("="*60)
print(f"\nAll outputs saved to: {OUT_DIR}")
print("  totalseg_dice_results.csv")
print("  totalseg_dice_chart.png")
print("  totalseg_coverage_chart.png")
print("  totalseg_overlay_comparison.png")