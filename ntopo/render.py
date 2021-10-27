
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import measure

from ntopo.utils import get_grid_points, get_grid_centers, get_default_figure_size


def save_densities_to_file(densities_np, filename):
    inv_gray = 1.0 - np.flipud(densities_np)
    image_np = np.tile(inv_gray[:, :, np.newaxis], (1, 1, 3))
    image_np = (256.0 * image_np)
    image_np = np.minimum(255.1, np.maximum(0.0, image_np)).astype(np.uint8)
    image = Image.fromarray(image_np)
    image.save(filename)


def plot_displacement_2d(disp_model, domain, nx, ny, save_path, save_prefix, save_postfix):
    points = get_grid_points(domain, [nx, ny])
    x_undef = np.reshape(points, (ny, nx, 2))
    displacements = disp_model.predict(points, batch_size=1000)
    displacements = np.reshape(displacements,  (ny, nx, 2))
    positions = x_undef + displacements
    positions_T = np.transpose(positions, axes=[1, 0, 2])

    fig = plt.figure(figsize=get_default_figure_size(domain), dpi=300)

    plt.plot(positions_T[:, :, 0], positions_T[:, :, 1], 'k', linewidth=0.5)
    plt.plot(positions[:, :, 0], positions[:, :, 1], 'k', linewidth=0.5)

    filename = os.path.join(save_path, save_prefix +
                            'displacement' + save_postfix + '.png')
    plt.savefig(filename)
    plt.close(fig)

def predict_densities_image_2d(density_model, domain, mirror, n_samples):
    [nx, ny] = n_samples
    positions = get_grid_centers(domain, n_samples)

    densities = density_model.predict(positions, batch_size=positions.shape[0] // 100)
    densities_im = np.reshape(densities, (ny, nx))
    domain_mirror = [domain[i] for i in range(len(domain))]
    
    if mirror[0]:
        right_part = np.fliplr(densities_im)
        densities_im = np.reshape(
            np.hstack((densities_im, right_part)), (ny, 2*nx))
        domain_mirror[1] = domain[0] + 2 * (domain[1] - domain[0])
    if mirror[1]:
        top_part = np.flipud(densities_im)
        densities_im = np.reshape(
            np.vstack((top_part, densities_im)), (2 * ny, nx))
        domain_mirror[3] = domain[2] + 2 * (domain[3] - domain[2])
    
    return densities_im, domain_mirror


def plot_densities_2d(density_model, domain, mirror, nx, ny, save_path, save_prefix, save_postfix):
    filename = os.path.join(save_path, save_prefix +
                            'density' + save_postfix + '.png')

    densities_im, _ = predict_densities_image_2d(density_model, domain, mirror, [nx, ny])
    print(f"mean: {np.mean(densities_im)}, max: {np.max(densities_im)}, min: {np.min(densities_im)}, ")

    save_densities_to_file(densities_im, filename)
    return filename


def save_densities_as_points_obj(densities, positions, iso_level, filename='', only_solid=True):

    if only_solid:
        solid = np.where(densities > iso_level)[0]

        obj_x = positions[solid, 0]
        obj_y = positions[solid, 1]
        obj_z = positions[solid, 2]
        solid_rho = densities[solid]
    else:
        obj_x = positions[:, 0]
        obj_y = positions[:, 1]
        obj_z = positions[:, 2]
        solid_rho = densities

    mapper = cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="jet")

    rho_color_map = mapper.to_rgba(solid_rho)
    rho_color_map = np.reshape(rho_color_map, (solid_rho.size, 4))
    plt.close()

    f = open(filename, "w")
    for i in range(len(obj_x)):
        f.write("v " + str(obj_x[i]) + " " + str(obj_y[i]) + " " + str(obj_z[i])
                + " " + str(rho_color_map[i, 0] * 255.0) +
                " " + str(rho_color_map[i, 1] * 255.0)
                + " " + str(rho_color_map[i, 2] * 255.0) + "\n")
    f.close()

def pad_with_zeros(density_grid):

    c0 = np.full(
        (1, density_grid.shape[1], density_grid.shape[2]), 0.0, dtype=np.float32)
    density_grid = np.concatenate((c0, density_grid, c0), axis=0)
    c1 = np.full(
        (density_grid.shape[0], 1, density_grid.shape[2]), 0.0, dtype=np.float32)
    density_grid = np.concatenate((c1, density_grid, c1), axis=1)
    c2 = np.full(
        (density_grid.shape[0], density_grid.shape[1], 1), 0.0, dtype=np.float32)
    density_grid = np.concatenate((c2, density_grid, c2), axis=2)
    return density_grid

def save_density_iso_surface(density_grid, spacing, iso_level, filename):
    density_grid = pad_with_zeros(density_grid)

    if np.amax(density_grid) < iso_level or np.amin(density_grid) > iso_level:
        print('cannot save density grid cause the levelset is empty')
        return

    verts, faces, normals, values = measure.marching_cubes(density_grid,
        level=iso_level, spacing=spacing, gradient_direction="ascent", method='lewiner')

    with open(filename, 'w') as file:
        for item in verts:
            file.write(f"v {item[0]} {item[1]} {item[2]}\n")

        for item in normals:
            file.write(f"vn {item[0]} {item[1]} {item[2]}\n")

        for item in faces:
            idx_0 = item[0] + 1
            idx_1 = item[1] + 1
            idx_2 = item[2] + 1
            file.write(
                f"f {idx_0}//{idx_0} {idx_1}//{idx_1} {idx_2}//{idx_2}\n")
