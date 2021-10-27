
import os
import math
import functools
import numpy as np
import tensorflow as tf

from ntopo.physics import TopoptEnergyModel, DiracForce, DistributedForce, DiracForceQ, QStrategy
from ntopo.utils import (
    compute_domain_volume, get_default_figure_size, get_default_sample_counts, scale_domain,
    get_grid_centers, get_grid_centers_spacing, get_grid_points
)
from ntopo.sdf import SDFDisk, SDFRectangle
from ntopo.constraints import DensityConstraint, DensityConstraintAdd, DensityConstraintNone, DisplacementConstraint, DisplacementPoint, DisplacementDisk, one_densities_function, zero_densities_function
from ntopo.render import plot_displacement_2d, plot_densities_2d, save_density_iso_surface, save_densities_as_points_obj


class Problem2D:
    def __init__(self):
        self.mirror = [False, False]
        self.density_constraint = DensityConstraintNone()

        self.dim = 2
        self.initialized = False

        self.domain = None
        self.free_volume = None
        self.energy_model = None
        self.constraint_volume = None
        self.forcing = None

    def init(self):
        assert isinstance(self.domain, np.ndarray)
        assert np.float32 == self.domain.dtype

        self.domain_volume = compute_domain_volume(self.domain)
        print('domain_volume ', self.domain_volume)
        self.energy_model = TopoptEnergyModel(dim=self.dim, dtype=tf.float32)
        self.initialized = True

        if isinstance(self.density_constraint, DensityConstraintNone):
            self.free_volume = self.domain_volume
            self.constraint_volume = 0.0
        else:
            # we want that integral of density model (which outputs targetVolumeRatio at beginning)  is equal to targetVolume
            # that means at the beginning the output volume is  targetVolumeRatio * freeVolume + constraintVolume
            self.free_volume, self.constraint_volume = self.density_constraint.estimate_volume(
                self.domain)

    def get_energy_model(self):
        assert self.initialized
        return self.energy_model

    def compute_force_loss(self, disp_model, samples):
        assert self.initialized
        return self.forcing.compute_force_loss(disp_model, samples)

    def plot_displacement(self, disp_model, save_path, save_prefix, save_postfix):
        assert self.initialized
        [nx, ny] = get_default_sample_counts(self.domain, 40 * 80)
        plot_displacement_2d(disp_model, self.domain, nx,
                             ny, save_path, save_prefix, save_postfix)

    def has_density_constraint(self):
        return self.density_constraint.has_constraint()

    def plot_densities(self, density_model, save_path, save_prefix, save_postfix):
        [nx, ny] = get_default_sample_counts(self.domain, 500 * 250)
        plot_densities_2d(density_model, self.domain, self.mirror,
                          nx, ny, save_path, save_prefix, save_postfix)

    def fem_bcs(self, n_vertices_x, n_vertices_y, element_width, element_height, fixed, forces):
        raise Exception('not implemented')



def fix_left(x, dim):
    return tf.tile(x[0], [1, dim])


class Beam2D(Problem2D):

    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.5, 0.0, 0.5], dtype=np.float32)

        self.domain = domain
        self.bc = functools.partial(fix_left, dim=self.dim)
        self.forcing = DiracForce(
            position=[domain[1], domain[2]], force=[0.0, -0.0025])
        self.domain = domain

    def fem_bcs(self, n_vertices_x, n_vertices_y, element_width, element_height, fixed, forces):
        forces[0, -1, 1] = -0.0025
        fixed[:, 0, :] = True # fix left column

class Distributed2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.5, 0.0, 0.5], dtype=np.float32)

        self.domain = domain
        self.bc = functools.partial(fix_left, dim=self.dim)
        self.forcing = DistributedForce(
            domain=[0, 1.5, 0.5, 0.5],
            dim=self.dim,
            n_samples=[400, 1],
            force_function=lambda x: (tf.zeros_like(
                x[0]), -0.0025*tf.ones_like(x[1]))
        )
    def fem_bcs(self, n_vertices_x, n_vertices_y, element_width, element_height, fixed, forces):
        forces[-1, 0, 1] = -0.0025*element_width*0.5
        forces[-1, 1:(n_vertices_x - 1), 1] = -0.0025*element_width
        forces[-1, -1, 1] = -0.0025*element_width*0.5
        fixed[:, 0, :] = True # fix left column


def fix_left_fix_xmid(x):
    return tf.concat((x[0] * (x[0] - 1),  x[0]), axis=1)


class BridgeMirrorBC2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.0, 0.0, 0.5], dtype=np.float32)

        self.domain = domain
        self.bc = fix_left_fix_xmid
        self.forcing = DistributedForce(
            domain=[0, 1.0, 0.0, 0.0],
            dim=self.dim,
            n_samples=[400, 1],
            force_function=lambda x: (tf.zeros_like(
                x[0]), -0.0025*tf.ones_like(x[1]))
        )
        self.mirror[0] = True
    def fem_bcs(self, n_vertices_x, n_vertices_y, element_width, element_height, fixed, forces):
        forces[0, :, 1] = -0.0025*element_width
        fixed[:, 0, :] = True # fix left column
        fixed[:, -1, 0] = True


class LongBeamMirrorBC2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.0, 0.0, 0.5], dtype=np.float32)

        self.domain = domain
        self.bc = fix_left_fix_xmid
        self.forcing = DiracForce(position=[1.0, 0.0], force=[0.0, -0.0025])
        self.mirror[0] = True
    def fem_bcs(self, n_vertices_x, n_vertices_y, element_width, element_height, fixed, forces):
        forces[0, -1, 1] = -0.0025
        fixed[:, 0, :] = True # fix left column
        fixed[:, -1, 0] = True


def fix_length_factor(x, displacement_constraint, dim):
    return tf.tile(displacement_constraint.compute_length_factor(tf.concat(x, axis=1)), [1, dim])


class ThreeHoles2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.5, 0.0, 0.5], dtype=np.float32)

        self.density_constraint = DensityConstraintAdd([
            DensityConstraint(
                SDFDisk([0.075, 0.5 - 0.075], 0.075, inside=True), one_densities_function),
            DensityConstraint(
                SDFDisk([0.075, 0.5 - 0.075], 0.035, inside=True), zero_densities_function),
            DensityConstraint(
                SDFDisk([0.075, 0.0 + 0.075], 0.075, inside=True), one_densities_function),
            DensityConstraint(
                SDFDisk([0.075, 0.0 + 0.075], 0.035, inside=True), zero_densities_function),

            DensityConstraint(SDFDisk(
                [1.5 - 0.075, 0.0 + 0.075], 0.075, inside=True), one_densities_function),
            DensityConstraint(SDFDisk(
                [1.5 - 0.075, 0.0 + 0.075], 0.035, inside=True), zero_densities_function),
        ])
        displacement_constraint = DisplacementConstraint(domain, [
            DisplacementDisk([0.075, 0.5 - 0.075], 0.035),
            DisplacementDisk([0.075, 0.0 + 0.075], 0.035),
        ])

        self.domain = domain
        self.bc = functools.partial(
            fix_length_factor, displacement_constraint=displacement_constraint, dim=self.dim)
        self.forcing = DiracForce(
            position=[1.5 - 0.075, 0.0 + 0.075 - 0.035], force=[0.0, -0.001])


class HoleInTheMiddle2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.5, 0.0, 0.5], dtype=np.float32)

        self.density_constraint = DensityConstraintAdd([
            DensityConstraint(
                SDFDisk([0.75, 0.25], 0.2, inside=True), zero_densities_function),
        ])

        self.domain = domain
        self.bc = functools.partial(fix_left, dim=self.dim)
        self.forcing = DiracForce(
            position=[domain[1], domain[2]], force=[0.0, -0.0025])


class Michell2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 2.0, 0.0, 1.6], dtype=np.float32)

        displacement_constraint = DisplacementConstraint(domain, [
            DisplacementPoint([2, 0.4]),
            DisplacementPoint([2, 1.2]),
        ])

        self.domain = domain
        self.bc = functools.partial(
            fix_length_factor, displacement_constraint=displacement_constraint, dim=self.dim)
        self.forcing = DiracForce(position=[0.0, 0.8], force=[0.0, -0.0025])


def get_shifted_grid_constraint_list(radius, nx, ny, grid_domain):
    constraint_list = []
    for i in range(ny):
        for j in range(nx):
            lx = j / (nx - 1.0)
            x = (1.0 - lx) * grid_domain[0] + lx * grid_domain[1]
            ly = i / (ny - 1.0)
            y = (1.0 - ly) * grid_domain[2] + ly * grid_domain[3]
            constraint_list.append(DensityConstraint(
                SDFDisk([x, y], radius, inside=True), zero_densities_function))
    for i in range(ny-1):
        for j in range(nx-1):
            lx = (j + 0.5) / (nx - 1.0)
            x = (1.0 - lx) * grid_domain[0] + lx * grid_domain[1]
            ly = (i + 0.5) / (ny - 1.0)
            y = (1.0 - ly) * grid_domain[2] + ly * grid_domain[3]
            constraint_list.append(DensityConstraint(
                SDFDisk([x, y], radius, inside=True), zero_densities_function))
    return constraint_list


class BeamWithManyHoles2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.5, 0.0, 0.5], dtype=np.float32)

        self.density_constraint = DensityConstraintAdd(get_shifted_grid_constraint_list(
            radius=0.04, nx=5, ny=3, grid_domain=[0.2, 1.3, 0.08, 0.42]))

        self.domain = domain
        self.bc = functools.partial(fix_left, dim=self.dim)
        self.forcing = DiracForce(
            position=[domain[1], domain[2]], force=[0.0, -0.0025])


def Column2D_bc(x):
    return tf.concat((x[0] - 0.25, x[1]), axis=1)


class Column2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 0.25, 0.0, 1.0], dtype=np.float32)
        dim = 2

        self.domain = domain
        self.dim = dim
        self.bc = Column2D_bc
        self.forcing = DistributedForce(
            domain=[0, 0.25, 1.0, 1.0],
            dim=dim,
            n_samples=[400, 1],
            force_function=lambda x: (tf.zeros_like(
                x[0]), -0.0025*tf.ones_like(x[1]))
        )
        self.mirror[0] = True


def LShapedBeam2D_bc(x):
    return tf.tile(1.0 - x[1], [1, 2])


class LShapedBeam2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.0, 0.0, 1.0], dtype=np.float32)
        cutoff = 0.6
        rect = [1.0 - cutoff, 1.0, 1.0 - cutoff, 1.0]

        self.density_constraint = DensityConstraint(
            SDFRectangle(rect), zero_densities_function)

        self.domain = domain
        self.bc = LShapedBeam2D_bc
        self.forcing = DiracForce(
            position=[domain[1], 0.5 * (1.0 - cutoff) + 0.5 * 0.0], force=[0.0, -0.0025])


def HalfHoop2D_bc(x):
    return tf.concat((0.5 - x[0],  x[1]), axis=1)


class HalfHoop2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 0.5, 0.0, 1.0], dtype=np.float32)

        self.domain = domain
        self.bc = HalfHoop2D_bc
        self.forcing = DistributedForce(
            domain=[0, 0.5*math.pi, 0.0, 0.0],
            dim=self.dim,
            n_samples=[512, 1],
            force_function=lambda x: (tf.zeros_like(
                x[0]), -0.01 * tf.ones_like(x[1])),
            position_function=lambda x: (
                0.5 - 0.5 * tf.math.sin(x[0]), 0.5 + 0.5 * tf.math.cos(x[0])),
        )
        self.mirror[0] = True


class Wheel2D(Problem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.05, 0.0, 1.05], dtype=np.float32)
        center = [0.525, 0.525]
        center_radius = 0.05
        force_radius = 0.5
        outer_radius = 0.525
        inner_radius = 0.475
        force_scale = 0.05
        displacement_constraint = DisplacementConstraint(
            domain, [DisplacementDisk(center, center_radius)])
        self.density_constraint = DensityConstraint(
            SDFUnion([
                SDFDisk(center, center_radius),
                SDFIntersection([SDFDisk(center, radius=outer_radius), SDFDisk(
                    center, radius=inner_radius, inside=False)])
            ]), one_densities_function)

        positions = []
        forces = []
        for i in range(4):
            t = 2.0 * math.pi * i / 4.0
            positions.append([center[0] + force_radius * math.cos(t),
                             center[1] + force_radius * math.sin(t)])
            forces.append([- force_scale * math.sin(t),
                          force_scale * math.cos(t)])

        self.domain = domain
        self.bc = functools.partial(
            fix_length_factor, displacement_constraint=displacement_constraint, dim=self.dim)
        self.forcing = DiracForce(position=positions, force=forces)

    
def mirror_positions(positions, domain, mirror):
    """
    expects positions to have shape (nx, ny, nz, 3)
    positions are not really mirrrored but offset, such that
    they match the mirrored densities from mirror_densities
    """
    for d in range(3):
        if mirror[d]:
            mirrored_xs_grid = positions.copy()
            mirrored_xs_grid[:, :, :, d] = (domain[2 * d + 1] - domain[2 * d]) + positions[:, :, :, d]
            positions = np.concatenate((positions, mirrored_xs_grid), axis=d)
    return positions

def mirror_densities(densities, mirror):
    """
    expects densities to have shape (nx, ny, nz, 3)
    """
    nx, ny, nz = densities.shape
    if mirror[0]:
        densities = np.concatenate(
            (densities, densities[range(nx - 1, -1, -1), :, :]), axis=0)
    if mirror[1]:
        densities = np.concatenate(
            (densities, densities[:, range(ny - 1, -1, -1), :]), axis=1)
    if mirror[2]:
        densities = np.concatenate(
            (densities, densities[:, :, range(nz - 1, -1, -1)]), axis=2)
    return densities

class Problem3D:
    def __init__(self):
        self.mirror = [False,  False, False]
        self.density_constraint = DensityConstraintNone()

        self.dim = 3
        self.initialized = False

        self.domain = None
        self.domain_volume = None
        self.free_volume = None
        self.constraint_volume = None
        self.energy_model = None
        self.forcing = None

    def init(self):
        assert isinstance(self.domain, np.ndarray)
        assert np.float32 == self.domain.dtype

        self.domain_volume = compute_domain_volume(self.domain)
        print('domain_volume ', self.domain_volume)
        self.energy_model = TopoptEnergyModel(dim=self.dim, dtype=tf.float32)
        self.initialized = True

        if isinstance(self.density_constraint, DensityConstraintNone):
            self.free_volume = self.domain_volume
            self.constraint_volume = 0.0
        else:
            # we want that integral of density model (which outputs targetVolumeRatio at beginning)  is equal to targetVolume
            # that means at the beginning the output volume is  targetVolumeRatio * freeVolume + constraintVolume
            self.free_volume, self.constraint_volume = self.density_constraint.estimate_volume(
                self.domain)

    def get_energy_model(self):
        assert self.initialized
        return self.energy_model

    def compute_force_loss(self, disp_model, samples):
        assert self.initialized
        return self.forcing.compute_force_loss(disp_model, samples)

    def plot_displacement(self, disp_model, save_path, save_prefix, save_postfix):
        [nx, ny, nz] = get_default_sample_counts(self.domain, 50*50*50)
        xs = get_grid_points(self.domain, [nx, ny, nz])
        displacements = disp_model.predict(xs, batch_size=1024)
        positions = xs + displacements
        densities = np.ones((positions.shape[0], 1), dtype=positions.dtype)
        save_densities_as_points_obj(densities=densities, positions=positions,
            iso_level=0.5, filename=os.path.join(
            save_path, save_prefix + 'displacement' + save_postfix + '.obj'), only_solid=False)

    def plot_densities(self, density_model, save_path, save_prefix, save_postfix):
        [nx, ny, nz] = get_default_sample_counts(self.domain, 50*50*50)
        self.save_density_to_obj(nx, ny, nz, density_model, iso_level=0.25, filename=os.path.join(
            save_path, save_prefix + 'density' + save_postfix + '.obj'))
        self.save_density_iso_surface(nx, ny, nz, density_model, iso_level=0.25, filename=os.path.join(
            save_path, save_prefix + 'density-iso' + save_postfix + '.obj'))

    def save_density_to_obj(self, nx, ny, nz, density_model, iso_level, filename='', only_solid=True):
        densities, _, _, positions = self.compute_mirrored_densities(nx, ny, nz, density_model)
        save_densities_as_points_obj(densities=densities, positions=positions,
            iso_level=iso_level, filename=filename, only_solid=only_solid)
        
    def compute_mirrored_densities(self, nx, ny, nz, density_model):
        # compute cell centers, so we can mirror consistently
        point_spacing = get_grid_centers_spacing(self.domain, n_cells=[nx, ny, nz], dtype=np.float32)
        xs = get_grid_centers(self.domain, n_cells=[nx, ny, nz], dtype=np.float32)
        xs_grid = np.reshape(xs, (nx, ny, nz, 3))
        densities = density_model.predict(xs, batch_size=1024)
        density_grid = np.reshape(densities, (nx, ny, nz))
        xs_grid = mirror_positions(xs_grid, domain=self.domain, mirror=self.mirror)
        density_grid = mirror_densities(density_grid, mirror=self.mirror)
        for d in range(3):
            assert xs_grid.shape[d] == density_grid.shape[d]
        nx, ny, nz, dim = xs_grid.shape
        
        densities = np.reshape(density_grid, (nx*ny*nz, 1))
        positions = np.reshape(xs_grid, (nx*ny*nz, 3))

        return densities, density_grid, point_spacing, positions

    def save_density_iso_surface(self, nx, ny, nz, density_model, iso_level, filename):
        _, density_grid, point_spacing, _ = self.compute_mirrored_densities(
            nx, ny, nz, density_model)
        save_density_iso_surface(density_grid=density_grid, spacing=point_spacing, iso_level=iso_level, filename=filename)


def BeamMirrorBC3D_bc(x):
    return tf.concat((tf.tile(x[0], [1, 2]),  x[0]*(x[2]-0.25)), axis=1)


class BeamMirrorBC3D(Problem3D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.0, 0.0, 0.5, 0.0, 0.25], dtype=np.float32)

        self.domain = domain
        self.bc = BeamMirrorBC3D_bc
        self.forcing = DistributedForce(
            domain=[1.0, 1.0, 0.0, 0.0, 0.0, 0.25],
            dim=self.dim,
            n_samples=[1, 1, 400],
            force_function=lambda x: (tf.zeros_like(
                x[0]), -0.0005 * tf.ones_like(x[1]), tf.zeros_like(x[2])),
        )
        self.mirror = [False, False, True]


def BridgeMirrorBC3D_bc(x):
    return tf.concat((x[0]*(x[0] - 1),   x[0], x[0]*(x[2]-0.25)), axis=1)


class BridgeMirrorBC3D(Problem3D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.0, 0.0, 0.5, 0.0, 0.25], dtype=np.float32)
        dim = 3

        self.domain = domain
        self.dim = dim
        self.bc = BridgeMirrorBC3D_bc
        self.forcing = DistributedForce(
            domain=[0.0, 1.0, 0.0, 0.0, 0.0, 0.25],
            dim=dim,
            n_samples=[500, 1, 125],
            force_function=lambda x: (tf.zeros_like(
                x[0]), -0.0025 * tf.ones_like(x[1]), tf.zeros_like(x[2])),
        )
        self.mirror = [True, False, True]


problems = [Beam2D, Distributed2D, BridgeMirrorBC2D, LongBeamMirrorBC2D, ThreeHoles2D, HoleInTheMiddle2D, Michell2D, BeamWithManyHoles2D, Column2D, LShapedBeam2D, HalfHoop2D, Wheel2D,
            BeamMirrorBC3D, BridgeMirrorBC3D]


def get_problem_by_name(problem_name):
    for p in problems:
        name = p.__name__
        if name == problem_name:
            return p()
    raise Exception('Problem not found with name ' + problem_name)


class SpaceProblem2D:
    def __init__(self):
        self.mirror = [False, False]
        self.density_constraint = DensityConstraintNone()

        self.dim = 2
        self.initialized = False

        self.domain = None
        self.q_domain = None
        self.free_volume = None
        self.constraint_volume = None
        self.energy_model = None
        self.forcing = None

    def init(self):
        assert isinstance(self.domain, np.ndarray)
        assert np.float32 == self.domain.dtype

        assert isinstance(self.q_domain, np.ndarray)
        assert np.float32 == self.q_domain.dtype

        self.domain_volume = compute_domain_volume(self.domain)
        self.free_volume = self.domain_volume
        print('domain_volume ', self.domain_volume)
        self.energy_model = TopoptEnergyModel(dim=self.dim, dtype=tf.float32)
        self.initialized = True

    def get_energy_model(self):
        assert self.initialized
        return self.energy_model

    def compute_force_loss(self, disp_model_q, samples):
        assert self.initialized
        return self.forcing.compute_force_loss(disp_model_q, samples)

    def get_plot_samples(self, n_q_samples):
        return [np.array([[q_i]], dtype=np.float32) for q_i in np.linspace(self.q_domain[0], self.q_domain[1], num=n_q_samples)]

    def plot_displacement(self, disp_model, save_path, save_prefix, save_postfix, q):
        [nx, ny] = get_default_sample_counts(self.domain, 40 * 80)
        plot_displacement_2d(disp_model.get_model_partial_q(
            q), self.domain, nx, ny, save_path, save_prefix, save_postfix=save_postfix)

    def plot_densities(self, density_model, save_path, save_prefix, save_postfix, n_q_samples=10):
        [nx, ny] = get_default_sample_counts(self.domain, 500 * 250)
        qs = self.get_plot_samples(n_q_samples)
        filenames = []
        for i in range(len(qs)):
            q = qs[i]
            assert np.size(q) == 1
            q0 = q.flatten()[0]
            save_postfix_i = f'{save_postfix}-q={q0:.6f}'
            fn = plot_densities_2d(density_model.get_model_partial_q(q), self.domain, self.mirror,
                                   nx, ny, save_path, save_prefix, save_postfix_i)
            filenames.append(fn)
        return filenames

    def fem_bcs(self, n_vertices_x, n_vertices_y, element_width, element_height, fixed, forces, q):
        raise Exception('fem_bcs: not implemented')


class Beam2DSpaceForce(SpaceProblem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.5, 0.0, 0.5], dtype=np.float32)
        q_domain = np.array([domain[2], domain[3]], dtype=np.float32)

        self.q_dim = 1
        self.domain = domain
        self.q_domain = q_domain
        self.volume_ratio_q_idx = -1
        self.bc = functools.partial(fix_left, dim=self.dim)
        self.forcing = DiracForceQ(
            position_function=lambda q: tf.concat(
                (domain[1] * tf.ones_like(q[0:1, :]), q[0:1, :], q[0:1, :]), axis=1),
            force_function=lambda q: tf.concat(
                (tf.zeros_like(q[0:1, :]), -0.0025 * tf.ones_like(q[0:1, :])), axis=1),
            dim=self.dim
        )
    def fem_bcs(self, n_vertices_x, n_vertices_y, element_width, element_height, fixed, forces, q):
        assert np.size(q) == 1
        q0 = q.item()
        idx0 = max(0, min(n_vertices_y - 2, math.floor(q0 / element_height) ))
        idx1 = idx0 + 1
        lam = max(0.0, min(1.0, (q - idx0 * element_height) / element_height ) )
        forces[idx0, -1, 1] = (1.0 - lam) *-0.0025
        forces[idx1, -1, 1] = lam * -0.0025
        fixed[:, 0, :] = True # fix left column


class Beam2DSpaceVolume(SpaceProblem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.5, 0.0, 0.5], dtype=np.float32)
        q_domain = np.array([0.3, 0.7], dtype=np.float32)

        self.q_dim = 1
        self.domain = domain
        self.q_domain = q_domain
        self.volume_ratio_q_idx = 0
        self.bc = functools.partial(fix_left, dim=self.dim)
        self.pos = tf.constant([[domain[1], domain[2]]], dtype=tf.float32)
        self.force = tf.constant([[0.0, -0.0025]], dtype=tf.float32)
        self.forcing = DiracForceQ(
            position_function=lambda q: tf.concat(
                (self.pos, q[0:1,  :]), axis=1),
            force_function=lambda q: self.force,
            dim=self.dim
        )
    def fem_bcs(self, n_vertices_x, n_vertices_y, element_width, element_height, fixed, forces, q):
        forces[0, -1, 1] = -0.0025
        fixed[:, 0, :] = True # fix left column


class BridgeMirrorBC2DSpaceVolume(SpaceProblem2D):
    def __init__(self):
        super().__init__()

        domain = np.array([0, 1.0, 0.0, 0.5], dtype=np.float32)
        q_domain = np.array([0.3, 0.7], dtype=np.float32)

        self.domain = domain
        self.q_dim = 1
        self.domain = domain
        self.q_domain = q_domain
        self.volume_ratio_q_idx = 0
        self.bc = fix_left_fix_xmid
        self.forcing = DistributedForce(
            domain=[0, 1.0, 0.0, 0.0],
            dim=self.dim,
            n_samples=[400, 1],
            force_function=lambda x: (tf.zeros_like(
                x[0]), -0.0025*tf.ones_like(x[1])),
            q_dim=self.q_dim,
            q_strategy=QStrategy.first,
        )
        self.mirror = [True, False]

    def fem_bcs(self, n_vertices_x, n_vertices_y, element_width, element_height, fixed, forces, q):
        fixed[:, 0, :] = True
        fixed[:, -1, 0] = True
        forces[0, :, 1] = -0.0025 * element_width

space_problems = [Beam2DSpaceForce,
                  Beam2DSpaceVolume, BridgeMirrorBC2DSpaceVolume]


def get_space_problem_by_name(problem_name):
    for p in space_problems:
        name = p.__name__
        if name == problem_name:
            return p()
    raise Exception('Problem not found with name ' + problem_name)
