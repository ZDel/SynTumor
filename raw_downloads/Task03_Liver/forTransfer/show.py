import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# === Load the label and image ===
label_path = "/home/zdelaney/SyntheticTumors/raw_downloads/Task03_Liver/forTransfer/label/lits-liver_32.nii.gz"
label_data = nib.load(label_path).get_fdata()

image_path = "/home/zdelaney/SyntheticTumors/raw_downloads/Task03_Liver/imagesTr/liver_32.nii.gz"
image_data = nib.load(image_path).get_fdata()

print(f"Image shape: {image_data.shape}")
print(f"Label shape: {label_data.shape}")

# === Find slices with tumor ===
tumor_slices = np.any(label_data == 2, axis=(0, 1))
if not np.any(tumor_slices):
    print("?? No tumor found in this label!")
else:
    slice_idx = np.where(tumor_slices)[0][0]
    print(f"? Displaying slice with tumor: {slice_idx}")

    # Extract slices
    image_slice = image_data[:, :, slice_idx]
    label_slice = label_data[:, :, slice_idx]

    # Create an RGB overlay
    overlay = np.zeros((label_slice.shape[0], label_slice.shape[1], 4), dtype=np.float32)

    # Tumor = green
    overlay[label_slice == 2, 1] = 1.0
    overlay[label_slice == 2, 3] = 0.5  # alpha

    # Liver = blue
    overlay[label_slice == 1, 2] = 1.0
    overlay[label_slice == 1, 3] = 0.5  # alpha

    # === Plot ===
    plt.figure(figsize=(6, 6))
    plt.imshow(image_slice.T, cmap='gray', origin='lower')
    plt.imshow(overlay.transpose(1, 0, 2), origin='lower')
    plt.title(f"CT with Liver (Blue) & Tumor (Green) Overlay - Slice {slice_idx}")
    plt.axis('off')
    plt.show()
