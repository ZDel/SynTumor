import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data import Dataset

# Import your SyntheticTumors transform
from TumorGenerated import TumorGenerated

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "/home/zdelaney/SyntheticTumors/raw_downloads/Task03_Liver/imagesTr/liver_119.nii.gz"
LABEL_PATH = "/home/zdelaney/SyntheticTumors/raw_downloads/Task03_Liver/labelsTr/liver_119.nii.gz"
OUTPUT_DIR = "./synthetic_grid_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# HELPER: Convert to numpy
# -----------------------------
def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else x

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def generate_synthetic_case(case_idx, grid_n=8):
    # -----------------------------
    # Load ORIGINAL CT/label
    # -----------------------------
    original_img_nii = nib.load(IMAGE_PATH)
    original_label_nii = nib.load(LABEL_PATH)
    original_img = original_img_nii.get_fdata()
    original_label = original_label_nii.get_fdata()
    affine = original_img_nii.affine

    # Choose center slice index
    slice_idx = original_img.shape[2] // 2

    orig_slice = original_img[:, :, slice_idx]
    orig_norm = (orig_slice - np.min(orig_slice)) / (np.max(orig_slice) - np.min(orig_slice))
    orig_norm = np.clip(orig_norm, 0, 1)

    # -----------------------------
    # Apply TumorGenerated Transform
    # -----------------------------
    data_list = [{"image": IMAGE_PATH, "label": LABEL_PATH}]
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        TumorGenerated(
            keys=["image", "label"],
            prob=1.0,
            tumor_prob=[0.0, 0.2, 0.4, 0.2, 0.2]
        )
    ])
    dataset = Dataset(data=data_list, transform=transforms)

    data = dataset[0]
    synthetic_image = to_numpy(data["image"][0])
    synthetic_label = to_numpy(data["label"][0])

    # -----------------------------
    # SAVE 3D NIfTI VOLUMES
    # -----------------------------
    img_out_path = os.path.join(OUTPUT_DIR, f"synthetic_image_{case_idx}.nii.gz")
    label_out_path = os.path.join(OUTPUT_DIR, f"synthetic_label_{case_idx}.nii.gz")

    nib.save(nib.Nifti1Image(synthetic_image, affine), img_out_path)
    nib.save(nib.Nifti1Image(synthetic_label, affine), label_out_path)

    print(f"? Saved synthetic 3D NIfTI image: {img_out_path}")
    print(f"? Saved synthetic 3D NIfTI label: {label_out_path}")
    import matplotlib.cm as cm
    
    label_colors = {
        2:  (1.0, 0.0, 0.0),      # red
        3:  (0.0, 1.0, 0.0),      # green
        4:  (0.0, 0.0, 1.0),      # blue
        5:  (1.0, 1.0, 0.0),      # yellow
        6:  (1.0, 0.0, 1.0),      # magenta
        7:  (0.0, 1.0, 1.0),      # cyan
        8:  (1.0, 0.5, 0.0),      # orange
        9:  (0.5, 0.0, 1.0),      # purple
        10: (0.0, 0.5, 1.0),      # light blue
        11: (0.5, 1.0, 0.0),      # lime
        12: (1.0, 0.0, 0.5),      # pink
        13: (0.0, 1.0, 0.5),      # teal
        14: (0.6, 0.6, 0.6),      # gray
    }
    fallback_cmap = cm.get_cmap('tab20')
    
    # ---------------------------
    # 1. Choose slice with most tumor
    # ---------------------------
    tumor_counts = np.sum(synthetic_label >= 2, axis=(0, 1))
    if np.all(tumor_counts == 0):
        print("[WARNING] No tumor found in label map!")
        slice_idx = synthetic_label.shape[2] // 2
    else:
        slice_idx = np.argmax(tumor_counts)
    
    # ---------------------------
    # 2. Extract chosen slice
    # ---------------------------
    syn_slice = synthetic_image[:, :, slice_idx]
    syn_label_slice = synthetic_label[:, :, slice_idx]
    
    # ---------------------------
    # 3. Normalize CT
    # ---------------------------
    syn_norm = (syn_slice - np.min(syn_slice)) / (np.max(syn_slice) - np.min(syn_slice) + 1e-8)
    syn_norm = np.clip(syn_norm, 0, 1)
    
    # ---------------------------
    # 4. Create RGB overlay
    # ---------------------------
    overlay_rgb = np.zeros((syn_norm.shape[0], syn_norm.shape[1], 3), dtype=np.float32)
    overlay_rgb[..., 0] = syn_norm
    overlay_rgb[..., 1] = syn_norm
    overlay_rgb[..., 2] = syn_norm
    
    # Overlay *each tumor label* in its color
    unique_labels = np.unique(syn_label_slice)
    for lbl in unique_labels:
        if lbl < 2:
            continue  # skip background and liver
    
        # Pick color for this label
        if lbl in label_colors:
            color = label_colors[lbl]
        else:
            color = fallback_cmap((lbl - 2) % fallback_cmap.N)[:3]
    
        mask = syn_label_slice == lbl
        for c in range(3):
            overlay_rgb[..., c][mask] = color[c]
    
    # ---------------------------
    # 5. Plot and Save
    # ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_norm, cmap='gray')
    axes[0].set_title("Original CT Slice")
    axes[0].axis('off')
    
    axes[1].imshow(syn_norm, cmap='gray')
    axes[1].set_title(f"Synthetic CT (Slice {slice_idx})")
    axes[1].axis('off')
    
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Synthetic CT + Tumor Labels")
    axes[2].axis('off')
    
    plt.tight_layout()
    grid_out_path = os.path.join(OUTPUT_DIR, f"synthetic_grid_{case_idx}.png")
    plt.savefig(grid_out_path)
    plt.close()
    print(f"? Saved grid image: {grid_out_path}")
    # -----------------------------
    # LOAD BACK THE SAVED VOLUMES
    # -----------------------------
    syn_img = nib.load(img_out_path).get_fdata()
    syn_lbl = nib.load(label_out_path).get_fdata()


    
    # Use matplotlib colormap as fallback for more labels
    fallback_cmap = cm.get_cmap('tab20')
    
    all_slices_rgb = []
    for i in range(syn_img.shape[2]):
        slice_img = syn_img[:, :, i]
        slice_lbl = syn_lbl[:, :, i]
    
        # Skip if no tumor labels
        if not np.any(slice_lbl >= 2):
            continue
    
        # Normalize background image
        norm_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-8)
        norm_img = np.clip(norm_img, 0, 1)
    
        # Make RGB
        rgb = np.zeros((norm_img.shape[0], norm_img.shape[1], 3), dtype=np.float32)
        rgb[..., 0] = norm_img
        rgb[..., 1] = norm_img
        rgb[..., 2] = norm_img
    
        # Overlay *all* tumor labels with different colors
        unique_labels = np.unique(slice_lbl)
        for lbl in unique_labels:
            if lbl < 2:
                continue  # skip background and liver mask
    
            # Get color
            if lbl in label_colors:
                color = label_colors[lbl]
            else:
                color = fallback_cmap((lbl - 2) % fallback_cmap.N)[:3]
    
            mask = slice_lbl == lbl
            for c in range(3):
                rgb[..., c][mask] = color[c]
    
        all_slices_rgb.append(rgb)
    
    print(f"[INFO] Found {len(all_slices_rgb)} slices with tumor")
    
    # ? Check if *any* slices have tumor at all
    if not all_slices_rgb:
        print("[WARNING] No tumor slices found! Skipping grid creation.")
        return

    # -----------------------------
    # MAKE n x n GRID
    # -----------------------------
    num_slices = len(all_slices_rgb)
    max_grid = grid_n * grid_n
    selected_slices = all_slices_rgb[:max_grid]
    if len(selected_slices) < max_grid:
        # Pad with blank images if needed
        H, W, _ = selected_slices[0].shape
        blank = np.zeros((H, W, 3), dtype=np.float32)
        selected_slices += [blank] * (max_grid - len(selected_slices))

    # Tile them
    rows = []
    for i in range(grid_n):
        row = np.hstack(selected_slices[i*grid_n:(i+1)*grid_n])
        rows.append(row)
    grid_image = np.vstack(rows)

    # -----------------------------
    # SAVE FINAL BIG GRID PNG
    # -----------------------------
    big_grid_out_path = os.path.join(OUTPUT_DIR, f"synthetic_all_slices_grid_{case_idx}.png")
    plt.imsave(big_grid_out_path, grid_image)
    print(f"? Saved ALL-slices grid image: {big_grid_out_path}")
# -----------------------------
# GENERATE MULTIPLE VARIANTS
# -----------------------------
NUM_VARIANTS = 20

for i in range(1, NUM_VARIANTS + 1):
    print(f"\n=== Generating synthetic case {i} ===")
    generate_synthetic_case(i)
