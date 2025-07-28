import nibabel as nib
import numpy as np
import os

input_folder = 'labelsTr'
output_folder = 'tumorMasks'

os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if fname.endswith('.nii.gz') and not fname.startswith('._'):
        print(f'Processing {fname}')
        # Load the label image
        label_img = nib.load(os.path.join(input_folder, fname))
        label_data = label_img.get_fdata()

        # Create tumor-only mask (2 -> 1, all else -> 0)
        tumor_mask = (label_data == 2).astype(np.uint8)

        # Save new NIfTI
        tumor_img = nib.Nifti1Image(tumor_mask, label_img.affine, label_img.header)
        nib.save(tumor_img, os.path.join(output_folder, fname))
