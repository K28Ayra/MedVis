import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import os

# ─── PATHS ───────────────────────────────────────────────────────────────────
IMAGES_DIR  = os.path.expanduser("~/Downloads/iiith/images")
LABELS_DIR  = os.path.expanduser("~/Downloads/iiith/labels")
PRED_DIR    = os.path.expanduser("~/Downloads/iiith/synthseg_output")
OUT_DIR     = os.path.expanduser("~/Downloads/iiith/results")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── LABEL NAMES (SynthSeg uses FreeSurfer IDs) ───────────────────────────────
LABEL_NAMES = {
    2:  "L Cerebral WM",      3:  "L Cerebral Cortex",
    4:  "L Lateral Ventricle",7:  "L Cerebellum WM",
    8:  "L Cerebellum Cortex",10: "L Thalamus",
    11: "L Caudate",          12: "L Putamen",
    13: "L Pallidum",         14: "3rd Ventricle",
    15: "4th Ventricle",      16: "Brain Stem",
    17: "L Hippocampus",      18: "L Amygdala",
    26: "L Accumbens",        28: "L Ventral DC",
    41: "R Cerebral WM",      42: "R Cerebral Cortex",
    43: "R Lateral Ventricle",46: "R Cerebellum WM",
    47: "R Cerebellum Cortex",49: "R Thalamus",
    50: "R Caudate",          51: "R Putamen",
    52: "R Pallidum",         53: "R Hippocampus",
    54: "R Amygdala",         58: "R Accumbens",
    60: "R Ventral DC",
}

LABEL_COLORS = {
    2:'#4a90d9', 3:'#5bc8af', 4:'#e87040', 7:'#3060b0',
    8:'#45a8e0', 10:'#e8c840', 11:'#e05080', 12:'#d080e0',
    13:'#50c840', 14:'#e04040', 15:'#40c0e0', 16:'#e8a040',
    17:'#c05050', 18:'#8050d0', 26:'#608030', 28:'#a08050',
    41:'#2878c8', 42:'#3aaa95', 43:'#d06030', 46:'#2050a0',
    47:'#35a0c0', 49:'#c8a830', 50:'#c04070', 51:'#b070d0',
    52:'#40b030', 53:'#a04040', 54:'#6040c0', 58:'#508020',
    60:'#806040',
}

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def dice_score(pred, gt, label_id):
    p = (pred == label_id)
    g = (gt == label_id)
    denom = p.sum() + g.sum()
    if denom == 0:
        return None
    return float(2 * (p & g).sum() / denom)

def make_overlay(img_sl, seg_sl, alpha=0.55):
    n = (img_sl - img_sl.min()) / (img_sl.max() - img_sl.min() + 1e-8)
    rgb = np.stack([n]*3, axis=-1)
    for lid, col in LABEL_COLORS.items():
        mask = seg_sl == lid
        if not mask.any(): continue
        r,g,b = int(col[1:3],16)/255, int(col[3:5],16)/255, int(col[5:7],16)/255
        rgb[mask] = (1-alpha)*rgb[mask] + alpha*np.array([r,g,b])
    return rgb

# ─── STEP 1: COMPUTE DICE SCORES ─────────────────────────────────────────────
print("Computing Dice scores...")
subjects = sorted([
    f.replace("_0000_synthseg.nii.gz","")
    for f in os.listdir(PRED_DIR)
    if f.endswith("_synthseg.nii.gz")
])

all_results = []
for subj in subjects:
    pred_path = os.path.join(PRED_DIR, f"{subj}_0000_synthseg.nii.gz")
    gt_path   = os.path.join(LABELS_DIR, f"{subj}_0000.nii.gz")

    if not os.path.exists(gt_path):
        print(f"  No GT for {subj}, skipping")
        continue

    pred_vol = nib.load(pred_path).get_fdata().astype(int)
    gt_vol   = nib.load(gt_path).get_fdata().astype(int)

    print(f"  {subj}: GT labels={np.unique(gt_vol).tolist()[:8]}...")

    for lid, name in LABEL_NAMES.items():
        dsc = dice_score(pred_vol, gt_vol, lid)
        if dsc is not None:
            all_results.append({
                "subject": subj, "label_id": lid,
                "structure": name, "DSC": round(dsc, 4)
            })

df = pd.DataFrame(all_results)
df.to_csv(os.path.join(OUT_DIR, "synthseg_dice_results.csv"), index=False)

mean_dsc = df["DSC"].mean()
print(f"\nOverall Mean DSC: {mean_dsc:.4f}")
print(f"Saved: {OUT_DIR}/synthseg_dice_results.csv")

# ─── STEP 2: THREE-PLANE VISUALIZATION ───────────────────────────────────────
print("\nGenerating visualizations...")
subj = subjects[0]
img  = nib.load(os.path.join(IMAGES_DIR,  f"{subj}_0000.nii.gz")).get_fdata()
gt   = nib.load(os.path.join(LABELS_DIR,  f"{subj}_0000.nii.gz")).get_fdata().astype(int)
pred = nib.load(os.path.join(PRED_DIR,    f"{subj}_0000_synthseg.nii.gz")).get_fdata().astype(int)

fig, axes = plt.subplots(3, 3, figsize=(15, 13), facecolor='#0d0d0d')
fig.suptitle(f'SynthSeg Brain Segmentation — {subj}', color='white', fontsize=15, fontweight='bold')

planes = [
    ("Sagittal", img[img.shape[0]//2,:,:], gt[gt.shape[0]//2,:,:], pred[pred.shape[0]//2,:,:]),
    ("Coronal",  img[:,img.shape[1]//2,:], gt[:,gt.shape[1]//2,:], pred[:,pred.shape[1]//2,:]),
    ("Axial",    img[:,:,img.shape[2]//2], gt[:,:,gt.shape[2]//2], pred[:,:,pred.shape[2]//2]),
]

for col, title in enumerate(['T1 MRI', 'Ground Truth', 'SynthSeg Prediction']):
    axes[0,col].set_title(title, color='white', fontsize=11, pad=8)

for row, (pname, ims, gts, ps) in enumerate(planes):
    axes[row,0].set_ylabel(pname, color='white', fontsize=10, rotation=0, labelpad=45, va='center')
    n = (ims - ims.min()) / (ims.max() - ims.min() + 1e-8)
    axes[row,0].imshow(n.T, cmap='gray', origin='lower', aspect='auto')
    axes[row,1].imshow(make_overlay(ims,gts).transpose(1,0,2), origin='lower', aspect='auto')
    axes[row,2].imshow(make_overlay(ims,ps).transpose(1,0,2), origin='lower', aspect='auto')
    for col in range(3):
        axes[row,col].axis('off')
        axes[row,col].set_facecolor('black')

plt.tight_layout()
out = os.path.join(OUT_DIR, f"{subj}_three_plane.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"Saved: {out}")

# ─── STEP 3: DICE BAR CHART ───────────────────────────────────────────────────
mean_per_struct = df.groupby(["label_id","structure"])["DSC"].mean().reset_index()
mean_per_struct = mean_per_struct.sort_values("DSC", ascending=True)

colors = ['#e05050' if v<0.5 else '#e8a040' if v<0.75 else '#5bc8af'
          for v in mean_per_struct["DSC"]]

fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0d0d0d')
ax.set_facecolor('#111111')
bars = ax.barh(mean_per_struct["structure"], mean_per_struct["DSC"],
               color=colors, height=0.7)

for bar, val in zip(bars, mean_per_struct["DSC"]):
    ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, color='white')

ax.axvline(x=mean_dsc, color='white', lw=1.5, label=f'Mean={mean_dsc:.3f}')
ax.axvline(x=0.85, color='#e8a040', lw=1, ls='--', alpha=0.5)
ax.axvline(x=0.50, color='#e05050', lw=1, ls='--', alpha=0.5)
ax.set_xlim(0, 1.1)
ax.set_xlabel('Dice Score (DSC)', color='white', fontsize=12)
ax.set_title(f'SynthSeg Per-Structure DSC  |  Mean = {mean_dsc:.4f}',
             color='white', fontsize=13, pad=10)
ax.tick_params(colors='white', labelsize=9)
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
out = os.path.join(OUT_DIR, "synthseg_dice_chart.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"Saved: {out}")

# ─── STEP 4: ERROR MAP ────────────────────────────────────────────────────────
subj = subjects[0]
z = img.shape[2] // 2
ims, gts_sl, ps_sl = img[:,:,z], gt[:,:,z], pred[:,:,z]
n = (ims - ims.min()) / (ims.max() - ims.min() + 1e-8)

rgb = np.stack([n]*3, axis=-1)
rgb[((gts_sl == ps_sl) & (gts_sl > 0))] = [0.2, 0.8, 0.3]
rgb[((gts_sl == 0)     & (ps_sl > 0))]  = [0.9, 0.2, 0.2]
rgb[((gts_sl > 0)      & (ps_sl == 0))] = [0.2, 0.4, 0.9]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0d0d0d')
fig.suptitle(f'Error Map — {subj}', color='white', fontsize=13)

axes[0].imshow(n.T, cmap='gray', origin='lower', aspect='auto')
axes[0].set_title('T1 MRI', color='white', fontsize=11)
axes[1].imshow(make_overlay(ims,gts_sl).transpose(1,0,2), origin='lower', aspect='auto')
axes[1].set_title('Ground Truth', color='white', fontsize=11)
axes[2].imshow(rgb.transpose(1,0,2), origin='lower', aspect='auto')
axes[2].set_title('Error Map', color='white', fontsize=11)

patches = [
    mpatches.Patch(color='#33cc4d', label='Correct (TP)'),
    mpatches.Patch(color='#e63333', label='False Positive'),
    mpatches.Patch(color='#3366e6', label='False Negative'),
]
axes[2].legend(handles=patches, facecolor='#1a1a1a', edgecolor='#333',
               labelcolor='white', fontsize=9)

for ax in axes:
    ax.axis('off')
    ax.set_facecolor('black')

plt.tight_layout()
out = os.path.join(OUT_DIR, "synthseg_error_map.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"Saved: {out}")

# ─── STEP 5: PER-SUBJECT COMPARISON ──────────────────────────────────────────
per_subj = df.groupby("subject")["DSC"].mean().reset_index()
per_subj.columns = ["Subject", "Mean DSC"]
per_subj = per_subj.sort_values("Mean DSC", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0d0d0d')
ax.set_facecolor('#111111')
colors_s = ['#5bc8af' if v >= 0.75 else '#e8a040' if v >= 0.5 else '#e05050'
            for v in per_subj["Mean DSC"]]
bars = ax.barh(per_subj["Subject"], per_subj["Mean DSC"], color=colors_s, height=0.5)

for bar, val in zip(bars, per_subj["Mean DSC"]):
    ax.text(val+0.003, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10, color='white')

ax.axvline(x=per_subj["Mean DSC"].mean(), color='white', lw=1.5,
           label=f'Mean={per_subj["Mean DSC"].mean():.3f}')
ax.set_xlim(0, 1.1)
ax.set_xlabel('Mean DSC', color='white', fontsize=12)
ax.set_title('Per-Subject Mean DSC — SynthSeg', color='white', fontsize=13)
ax.tick_params(colors='white', labelsize=10)
for sp in ax.spines.values(): sp.set_edgecolor('#333')
ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='white', fontsize=9)

plt.tight_layout()
out = os.path.join(OUT_DIR, "synthseg_per_subject.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"Saved: {out}")

# ─── STEP 6: AXIAL SLICES ─────────────────────────────────────────────────────
n_slices = 6
offsets = np.linspace(-30, 30, n_slices, dtype=int)
z_mid = img.shape[2] // 2

fig, axes = plt.subplots(3, n_slices, figsize=(18, 8), facecolor='#0d0d0d')
fig.suptitle(f'Axial Slices — {subj}', color='white', fontsize=13)

for row, label in enumerate(['T1 MRI', 'Ground Truth', 'SynthSeg']):
    axes[row,0].set_ylabel(label, color='white', fontsize=9, rotation=90, labelpad=8)

for col, off in enumerate(offsets):
    z = int(z_mid + off)
    n_sl = (img[:,:,z] - img[:,:,z].min()) / (img[:,:,z].max() - img[:,:,z].min() + 1e-8)
    axes[0,col].imshow(n_sl.T, cmap='gray', origin='lower', aspect='auto')
    axes[0,col].set_title(f'z={z}', color='white', fontsize=8)
    axes[1,col].imshow(make_overlay(img[:,:,z], gt[:,:,z]).transpose(1,0,2),
                       origin='lower', aspect='auto')
    axes[2,col].imshow(make_overlay(img[:,:,z], pred[:,:,z]).transpose(1,0,2),
                       origin='lower', aspect='auto')
    for row in range(3):
        axes[row,col].axis('off')
        axes[row,col].set_facecolor('black')

plt.tight_layout()
out = os.path.join(OUT_DIR, "synthseg_axial_slices.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()
print(f"Saved: {out}")

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("SYNTHSEG RESULTS SUMMARY")
print("="*55)
print(f"Subjects evaluated:    {df['subject'].nunique()}")
print(f"Structures per subject:{df['label_id'].nunique()}")
print(f"Overall Mean DSC:      {mean_dsc:.4f}")
print(f"Best structure:        {df.loc[df['DSC'].idxmax(), 'structure']} "
      f"({df['DSC'].max():.4f})")
print(f"Worst structure:       {df.loc[df['DSC'].idxmin(), 'structure']} "
      f"({df['DSC'].min():.4f})")
print("="*55)
print(f"\nAll outputs saved to: {OUT_DIR}")
print("  synthseg_dice_results.csv")
print("  synthseg_dice_chart.png")
print("  synthseg_three_plane.png")
print("  synthseg_error_map.png")
print("  synthseg_per_subject.png")
print("  synthseg_axial_slices.png")