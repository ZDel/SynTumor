import nibabel as nib
import numpy as np
import os

images_folder = 'imagesTr'
tumor_masks_folder = 'tumorMasks'
healthy_output_folder = 'healthy_images'
os.makedirs(healthy_output_folder, exist_ok=True)

for fname in os.listdir(images_folder):
    if not fname.endswith('.nii.gz') or fname.startswith('._'):
        continue

    print(f"Processing {fname}")
    # Load original CT
    img_nii = nib.load(os.path.join(images_folder, fname))
    img_data = img_nii.get_fdata()

    # Load tumor mask
    mask_nii = nib.load(os.path.join(tumor_masks_folder, fname))
    mask_data = mask_nii.get_fdata()

    # Mask out tumor (simple version: set to -1000 HU = air)
    healthy_data = np.copy(img_data)
    healthy_data[mask_data == 1] = -1000

    # Save new healthy liver
    healthy_img_nii = nib.Nifti1Image(healthy_data, img_nii.affine, img_nii.header)
    nib.save(healthy_img_nii, os.path.join(healthy_output_folder, fname))
