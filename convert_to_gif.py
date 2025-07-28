import os
import sys
import numpy as np
import nibabel as nib
import imageio
import argparse

def normalize_image(vol):
    vol = vol.astype(np.float32)
    vol = vol - np.min(vol)
    vol = vol / (np.max(vol) + 1e-8)
    vol = (vol * 255).astype(np.uint8)
    return vol

def nifti_to_gif(input_path, output_path, axis=2, fps=5):
    print(f"[INFO] Loading NIfTI: {input_path}")
    img = nib.load(input_path)
    data = img.get_fdata()
    
    print(f"[INFO] Data shape: {data.shape}")

    # Normalize
    data = normalize_image(data)

    # Make list of frames
    frames = []
    num_slices = data.shape[axis]
    print(f"[INFO] Generating {num_slices} frames along axis {axis}")
    start_slice = 150
    if start_slice >= num_slices:
        print(f"[WARNING] Requested to skip {start_slice} slices but only {num_slices} available!")
        start_slice = 0
    
    for i in range(start_slice, num_slices):
        if axis == 0:
            slice_img = data[i, :, :]
        elif axis == 1:
            slice_img = data[:, i, :]
        else:
            slice_img = data[:, :, i]
        frames.append(slice_img)
    
    print(f"[INFO] Saving GIF: {output_path}")
    imageio.mimsave(output_path, frames, duration=1/fps, loop=0)
    print("[DONE] GIF saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NIfTI to animated GIF")
    parser.add_argument("input", help="Path to input NIfTI file (.nii or .nii.gz)")
    parser.add_argument("output", help="Path to output GIF file")
    parser.add_argument("--axis", type=int, default=2, help="Axis to slice along (default=2)")
    parser.add_argument("--fps", type=float, default=5, help="Frames per second (default=5)")

    args = parser.parse_args()

    nifti_to_gif(args.input, args.output, axis=args.axis, fps=args.fps)

