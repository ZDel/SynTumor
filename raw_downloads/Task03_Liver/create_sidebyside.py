import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# === Folders ===
image_folder = 'imagesTr'
mask_folder = 'tumorMasks'
output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)

# === List all images ===
all_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.nii.gz') and not f.startswith('._')])

for fname in all_files:
    print(f"Processing {fname}")
    try:
        # Load CT and tumor mask
        image_nii = nib.load(os.path.join(image_folder, fname))
        mask_nii = nib.load(os.path.join(mask_folder, fname))

        image_data = image_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        # Find a slice with tumor
        tumor_slices = np.any(mask_data, axis=(0, 1))
        if not np.any(tumor_slices):
            print(f"?? No tumor found in {fname}. Skipping.")
            continue

        # Pick first slice with tumor
        slice_idx = np.where(tumor_slices)[0][0]
        print(f"  Tumor slice index: {slice_idx}")

        image_slice = image_data[:, :, slice_idx]
        mask_slice = mask_data[:, :, slice_idx]

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(image_slice.T, cmap='gray', origin='lower')
        axs[0].set_title('CT Image')
        axs[0].axis('off')

        axs[1].imshow(image_slice.T, cmap='gray', origin='lower')
        axs[1].imshow(mask_slice.T, cmap='Reds', alpha=0.5, origin='lower')
        axs[1].set_title('Tumor Overlay')
        axs[1].axis('off')

        plt.tight_layout()

        # Save figure
        out_path = os.path.join(output_folder, fname.replace('.nii.gz', '.png'))
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"? Saved: {out_path}")

    except Exception as e:
        print(f"? Error processing {fname}: {e}")
