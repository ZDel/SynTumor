import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# === Folder paths ===
base_dir = "/home/zdelaney/SyntheticTumors/raw_downloads/Task03_Liver/forTransfer"
image_dir = os.path.join(base_dir, "04_LiTS/image")  # one level up
label_dir = os.path.join(base_dir, "label")
output_dir = os.path.join(base_dir, "../tumorPNGs")

os.makedirs(output_dir, exist_ok=True)

# === List all label files ===
all_labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz') and not f.startswith('._')])

print(f"Found {len(all_labels)} label files.")

for fname in all_labels:
    print(f"\nProcessing: {fname}")
    try:
        # === Load label (tumor mask) ===
        label_nii = nib.load(os.path.join(label_dir, fname))
        label_data = label_nii.get_fdata()

        # === Load matching CT image ===
        image_path = os.path.join(image_dir, fname.replace('lits-', ''))
        if not os.path.exists(image_path):
            print(f"?? No matching CT image for {fname}. Skipping.")
            continue

        image_nii = nib.load(image_path)
        image_data = image_nii.get_fdata()

        # === Find slices with tumor ===
        tumor_slices = np.any(label_data, axis=(0, 1))
        if not np.any(tumor_slices):
            print("?? No tumor found in this mask. Skipping.")
            continue

        # === Pick first slice with tumor ===
        slice_idx = np.where(tumor_slices)[0][0]
        print(f"  Tumor found at slice {slice_idx}")

        image_slice = image_data[:, :, slice_idx]
        mask_slice = label_data[:, :, slice_idx]

        # === Plot side by side ===
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(image_slice.T, cmap='gray', origin='lower')
        axs[0].set_title('CT Image')
        axs[0].axis('off')

        axs[1].imshow(image_slice.T, cmap='gray', origin='lower')
        axs[1].imshow(mask_slice.T, cmap='Reds', alpha=0.5, origin='lower')
        axs[1].set_title('Tumor Overlay')
        axs[1].axis('off')

        plt.tight_layout()

        # === Save figure ===
        out_png = os.path.join(output_dir, fname.replace('.nii.gz', '.png'))
        plt.savefig(out_png, bbox_inches='tight')
        plt.close()

        print(f"? Saved: {out_png}")

    except Exception as e:
        print(f"? Error processing {fname}: {e}")
