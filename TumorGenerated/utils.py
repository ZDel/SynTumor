import random
import cv2
import elasticdeform
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

# -------------------------------
# Utility function to save axial slice as PNG
# -------------------------------
def save_slice_png(volume, z_index, out_path, title=None, cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(volume[..., z_index], cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# -------------------------------
# Generate probabilistic texture
# -------------------------------
def generate_prob_function(mask_shape):
    sigma = np.random.uniform(3, 15)
    a = np.random.uniform(0, 1, size=mask_shape)
    a_2 = gaussian_filter(a, sigma=sigma)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base
    return a

def get_texture(mask_shape, save_dir=None, case_id=None):
    a = generate_prob_function(mask_shape)
    random_sample = np.random.uniform(0, 1, size=mask_shape)
    b = (a > random_sample).astype(float)

    sigma_b = np.random.uniform(3, 5) if np.random.uniform() < 0.7 else np.random.uniform(5, 8)
    b2 = gaussian_filter(b, sigma_b)

    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b2 > 0.12
    beta = u_0 / (np.sum(b2 * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b2, 0, 1)

    if save_dir:
        z_center = Bj.shape[2] // 2
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{case_id}_step3_texture_addition.png")
        save_slice_png(Bj, z_center, out_path, title="Texture addition", cmap='viridis')

    return Bj

def get_predefined_texture(mask_shape, sigma_a, sigma_b):
    a = np.random.uniform(0, 1, size=mask_shape)
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    random_sample = np.random.uniform(0, 1, size=mask_shape)
    b = (a > random_sample).astype(float)
    b = gaussian_filter(b, sigma_b)

    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b, 0, 1)
    return Bj

# -------------------------------
# Target Location Selection
# -------------------------------
def random_select(mask_scan, image_scan=None,save_dir=None, case_id=None):
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]
    z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start
    liver_mask = mask_scan[..., z]

    kernel = np.ones((5, 5), dtype=np.uint8)
    liver_mask_eroded = cv2.erode(liver_mask, kernel, iterations=1)

    coordinates = np.argwhere(liver_mask_eroded == 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist()
    xyz.append(z)

    if save_dir:
        liver_mask_copy = liver_mask.copy()
        liver_mask_copy[xyz[0], xyz[1]] = 2
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{case_id}_step1_target_location.png")
        save_slice_png(liver_mask_copy[np.newaxis, :, :], 0, out_path, title="Target location", cmap='viridis')

    return xyz

# -------------------------------
# Ellipsoid Generation
# -------------------------------
def get_ellipsoid(x, y, z, save_dir=None, case_id=None):
    sh = (4 * x, 4 * y, 4 * z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2 * x, 2 * y, 2 * z])

    bboxl = np.floor(com - radii).clip(0, None).astype(int)
    bboxh = (np.ceil(com + radii) + 1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice, bboxl, bboxh))]
    roiaux = aux[tuple(map(slice, bboxl, bboxh))]

    logrid = *map(np.square, np.ogrid[tuple(
        map(slice, (bboxl - com) / radii, (bboxh - com - 1) / radii, 1j * (bboxh - bboxl)))]),
    dst = (1 - sum(logrid)).clip(0, None)
    mask = dst > roiaux
    roi[mask] = 1
    np.copyto(roiaux, dst, where=mask)

    if save_dir:
        z_center = out.shape[2] // 2
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{case_id}_step2_ellipsoid_raw.png")
        save_slice_png(out, z_center, out_path, title="Ellipsoid generation")

    return out

# -------------------------------
# Tumor Geometry Deformation
# -------------------------------
def get_fixed_geo(mask_scan, tumor_type, save_dir=None, case_id=None):
    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros(
        (mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z),
        dtype=np.int8
    )

    radius_dict = {
        'tiny': 4,
        'small': 8,
        'medium': 16,
        'large': 32
    }

    def generate_and_paste(num_tumor, base_radius, sigma_range):
        current_label = 2

        for i in range(num_tumor):
            x = random.randint(int(0.75 * base_radius), int(1.25 * base_radius))
            y = random.randint(int(0.75 * base_radius), int(1.25 * base_radius))
            z = random.randint(int(0.75 * base_radius), int(1.25 * base_radius))
            sigma = random.uniform(*sigma_range)

            geo = get_ellipsoid(x, y, z, save_dir=save_dir, case_id=f"{case_id}_tumor{i}")
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            
            # 2. Texture
            texture_box = get_texture(geo.shape, save_dir=None, case_id=None)
            masked_texture = geo * texture_box
            
            # 3. Save masked texture PNG
            if save_dir:
                z_center = masked_texture.shape[2] // 2
                out_path = os.path.join(save_dir, f"{case_id}_tumor{i}_step3_texture_addition.png")
                save_slice_png(masked_texture, z_center, out_path, title="Masked Texture Addition", cmap='viridis')
            
            # 4. Paste tumor mask
            point = random_select(mask_scan, save_dir=save_dir, case_id=f"{case_id}_tumor{i}")
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2
            
            geo_mask[
                      x_low:x_high, y_low:y_high, z_low:z_high
                  ][geo > 0] = current_label
            current_label += 1

    if tumor_type == "tiny":
        generate_and_paste(random.randint(3, 10), radius_dict['tiny'], (0.5, 1))
    elif tumor_type == "small":
        generate_and_paste(random.randint(3, 10), radius_dict['small'], (1, 2))
    elif tumor_type == "medium":
        generate_and_paste(random.randint(2, 5), radius_dict['medium'], (3, 6))
    elif tumor_type == "large":
        generate_and_paste(random.randint(1, 3), radius_dict['large'], (5, 10))
    elif tumor_type == "mix":
        generate_and_paste(random.randint(3, 10), radius_dict['tiny'], (0.5, 1))
        generate_and_paste(random.randint(5, 10), radius_dict['small'], (1, 2))
        generate_and_paste(random.randint(2, 5), radius_dict['medium'], (3, 6))
        generate_and_paste(random.randint(1, 3), radius_dict['large'], (5, 10))

    geo_mask = geo_mask[
        enlarge_x // 2:-enlarge_x // 2,
        enlarge_y // 2:-enlarge_y // 2,
        enlarge_z // 2:-enlarge_z // 2
    ]
    #geo_mask = (geo_mask * mask_scan) >= 1
    geo_mask = geo_mask*mask_scan
    if save_dir:
        z_center = geo_mask.shape[2] // 2
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{case_id}_step4_final_deformation.png")
        save_slice_png(geo_mask.astype(float), z_center, out_path, title="Final deformation")

    return geo_mask

# -------------------------------
# Tumor Insertion
# -------------------------------
def get_tumor(volume_scan, mask_scan, tumor_type, texture, save_dir=None, case_id=None):
    geo_mask = get_fixed_geo(mask_scan, tumor_type, save_dir=save_dir, case_id=case_id)

    sigma = np.random.uniform(1, 2)
    difference = np.random.uniform(65, 145)

    geo_blur = gaussian_filter(geo_mask * 1.0, sigma)
    abnormally = (volume_scan - texture * geo_blur * difference) * mask_scan
    abnormally_full = volume_scan * (1 - mask_scan) + abnormally
    abnormally_mask = mask_scan + geo_mask

    return abnormally_full, abnormally_mask

# -------------------------------
# High-Level Entry Point
# -------------------------------
def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture, save_dir=None, case_id=None):
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
    y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
    z_start, z_end = max(0, z_start + 1), min(mask_scan.shape[2], z_end - 1)

    liver_volume = volume_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]

    x_length, y_length, z_length = x_end - x_start, y_end - y_start, z_end - z_start
    start_x = random.randint(0, texture.shape[0] - x_length - 1)
    start_y = random.randint(0, texture.shape[1] - y_length - 1)
    start_z = random.randint(0, texture.shape[2] - z_length - 1)
    cut_texture = texture[start_x:start_x + x_length, start_y:start_y + y_length, start_z:start_z + z_length]

    liver_volume, liver_mask = get_tumor(
        liver_volume, liver_mask, tumor_type, cut_texture, save_dir=save_dir, case_id=case_id
    )

    volume_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_volume
    mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_mask

    return volume_scan, mask_scan
