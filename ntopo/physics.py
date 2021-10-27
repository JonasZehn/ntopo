
from enum import Enum

import tensorflow as tf
import numpy as np

from ntopo.utils import tf_to_np_data_type, stratified_sampling, compute_domain_volume


def compute_lin_elasticity_strain_energy_core_2d(
    du0dx0, du0dx1, du1dx0, du1dx1
):
    youngs_modulus = 1.0
    poissons_ratio = 0.3

    eps11 = du0dx0
    eps12 = 0.5 * du0dx1 + 0.5 * du1dx0
    #eps21 = eps12
    eps22 = du1dx1

    trace_strain = eps11 + eps22
    # see plane stress conditions: https://en.wikipedia.org/wiki/Hooke%27s_law#Plane_stress
    shear_modulus_times_2 = youngs_modulus / (1 + poissons_ratio)
    second_term_factor = youngs_modulus * \
        poissons_ratio / (1 - poissons_ratio*poissons_ratio)
    sigma11 = shear_modulus_times_2 * eps11 + second_term_factor * trace_strain
    sigma22 = shear_modulus_times_2 * eps22 + second_term_factor * trace_strain
    sigma12 = shear_modulus_times_2 * eps12
    #sigma21 = shear_modulus_times_2 * eps21

    # contraction: 0.5 * sigma : epsilon
    #energy = 0.5 * (eps11 * sigma11 + eps12 * sigma12 +
    #                eps21 * sigma21 + eps22 * sigma22)
    energy = 0.5 * (eps11 * sigma11 + 2.0 * eps12 * sigma12 + \
                    eps22 * sigma22)
    return energy


@tf.function()
def compute_lin_elasticity_strain_energy_core_2d_tf(
    du0dx0, du0dx1, du1dx0, du1dx1
):
    return compute_lin_elasticity_strain_energy_core_2d(du0dx0, du0dx1, du1dx0, du1dx1)


def compute_lin_elasticity_strain_energy_2d(dudx):
    """
    takes a list of tensors where each tensor i is n rows of jacobians du[i]/dx
    returns energy, one for each row of X
    """
    assert isinstance(dudx, list)
    assert len(dudx) == 2
    
    dudx_split = []
    for u_idx in range(2):
        for x_idx in range(2):
            dudx_k = tf.gather(dudx[u_idx], x_idx, axis=-1, batch_dims=0)
            dudx_split.append(dudx_k)

    energy_lin_elasticity = compute_lin_elasticity_strain_energy_core_2d_tf(
        dudx_split[0], dudx_split[1], dudx_split[2], dudx_split[3]
    )
    energy_lin_elasticity = tf.expand_dims(energy_lin_elasticity, axis=-1)

    return energy_lin_elasticity


def compute_lin_elasticity_strain_energy_core_3d(
    du0dx0, du0dx1, du0dx2, du1dx0, du1dx1, du1dx2, du2dx0, du2dx1, du2dx2
):
    youngs_modulus = 1.0
    poissons_ratio = 0.3
    lame_mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    lame_lambda = youngs_modulus * poissons_ratio / \
        ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))

    eps11 = du0dx0
    eps12 = 0.5 * du0dx1 + 0.5 * du1dx0
    eps13 = 0.5 * du0dx2 + 0.5 * du2dx0
    #eps21 = eps12
    eps22 = du1dx1
    eps23 = 0.5 * du1dx2 + 0.5 * du2dx1
    #eps31 = eps13
    #eps32 = eps23
    eps33 = du2dx2

    trace_strain = eps11 + eps22 + eps33
    squared_diagonal = eps11 * eps11 + eps33 * eps33 + eps22 * eps22
    energy = 0.5 * lame_lambda * trace_strain * trace_strain + lame_mu * \
        (squared_diagonal + 2.0 * eps12 * eps12 +
         2.0 * eps13 * eps13 + 2.0 * eps23 * eps23)
    return energy


@tf.function()
def compute_lin_elasticity_strain_energy_core_3d_tf(
    du0dx0, du0dx1, du0dx2, du1dx0, du1dx1, du1dx2, du2dx0, du2dx1, du2dx2
):
    return compute_lin_elasticity_strain_energy_core_3d(du0dx0, du0dx1, du0dx2, du1dx0, du1dx1, du1dx2, du2dx0, du2dx1, du2dx2)


def compute_lin_elasticity_strain_energy_3d(dudx):
    """
    takes a list of tensors where each tensor i is n rows of jacobians du[i]/dx
    returns energy, one for each row of X
    """
    assert isinstance(dudx, list)
    assert len(dudx) == 3
    
    dudx_split = []
    for u_idx in range(3):
        for x_idx in range(3):
            dudx_k = tf.gather(dudx[u_idx], x_idx, axis=1, batch_dims=0)
            dudx_split.append(dudx_k)

    energy_lin_elasticity = compute_lin_elasticity_strain_energy_core_3d_tf(
        dudx_split[0], dudx_split[1], dudx_split[2], dudx_split[3], dudx_split[4],
        dudx_split[5],dudx_split[6], dudx_split[7], dudx_split[8]
    )
    energy_lin_elasticity = tf.expand_dims(energy_lin_elasticity, axis=1)

    return energy_lin_elasticity


QStrategy = Enum('QStrategy', 'none first', module=__name__)


class Forcing:
    def __init__(self, dtype, dim):
        self.data_type_tf = dtype
        self.data_type_np = tf_to_np_data_type(dtype)
        self.dim = dim


class DiracForce(Forcing):
    def __init__(self, position, force, dtype=tf.float32):
        assert isinstance(force, list)
        if isinstance(force[0], list):
            super().__init__(dtype, dim=len(force[0]))

            self.num_forces = len(force)

            self.force = np.array(force, dtype=self.data_type_np)
            self.force_position = np.array(position, dtype=self.data_type_np)

            assert self.force_position.shape[0] == self.num_forces and self.force_position.shape[1] == self.dim
            assert self.force.shape[0] == self.num_forces and self.force.shape[1] == self.dim
            assert self.force.ndim == 2 and self.force_position.ndim == 2
        else:
            super().__init__(dtype, dim=np.array(force).size)

            self.num_forces = 1

            self.force = np.reshape(
                np.array(force, dtype=self.data_type_np),  (1, self.dim))
            self.force_position = np.reshape(
                np.array(position, dtype=self.data_type_np),  (1, self.dim))
            assert self.force_position.ndim == 2 and self.force_position.shape[1] == self.dim
            assert self.force.ndim == 2 and self.force.shape[1] == self.dim

    def compute_force_loss(self, disp_model, samples):
        pos = tf.constant(self.force_position, dtype=self.data_type_tf)
        displacements = disp_model(pos)
        force_tf = tf.constant(self.force, dtype=self.data_type_tf)
        result = - \
            tf.math.reduce_mean(tf.math.reduce_sum(displacements * force_tf,
                               axis=1, keepdims=True), axis=0, keepdims=True)
        return result


class DistributedForce(Forcing):
    def __init__(self, domain, dim, n_samples, force_function, position_function=None, dtype=tf.float32, q_dim=0, q_strategy=QStrategy.none):
        super().__init__(dtype, dim)

        self.dim = dim
        self.position_function = position_function
        self.force_function = force_function
        self.domain = np.copy(domain)
        self.volume = compute_domain_volume(domain)
        self.n_samples = n_samples

        self.q_dim = q_dim
        self.q_strategy = q_strategy

    def compute_force_loss(self, disp_model, samples):
        force_positions = stratified_sampling(
            self.domain, self.n_samples, n_points_per_cell=1, dtype=self.data_type_np)

        positions_split = []
        for dimension in range(self.dim):
            positions_split.append(tf.gather(force_positions, [dimension], axis=1))

        if self.position_function:
            positions_split =  list( self.position_function(positions_split) )
            force_positions = tf.concat(positions_split, axis=1)

        if self.q_strategy == QStrategy.none:
            displacements = disp_model(force_positions)
        else:
            assert self.q_strategy == QStrategy.first
            q = tf.gather(tf.gather(samples, [0], axis=0), tf.range(
                self.dim, (self.dim + self.q_dim)), axis=1)
            disp_inputs = tf.concat(
                (force_positions, tf.tile(q, tf.stack((tf.shape(force_positions)[0], 1)))), axis=1)
            displacements = disp_model(disp_inputs)

        # should return a touple describing the force
        force_tuple = self.force_function(positions_split)

        forces = tf.concat(force_tuple, axis=1)
        force_energies = - \
            tf.reduce_sum(forces * displacements, axis=1, keepdims=True)
        return self.volume * tf.reduce_mean(force_energies, keepdims=True)


class DiracForceQ(Forcing):
    def __init__(self, position_function, force_function, dim, dtype=tf.float32):
        super().__init__(dtype, dim=dim)

        self.position_function = position_function
        self.force_function = force_function
        self.num_forces = 1

    def compute_force_loss(self, disp_model, samples):
        q = tf.gather(samples, [self.dim], axis=1)
        pos = self.position_function(q)
        displacements = disp_model(pos)
        force_tf = self.force_function(q)
        result = - \
            tf.math.reduce_mean(tf.math.reduce_sum(displacements * force_tf,
                               axis=1, keepdims=True), axis=0, keepdims=True)
        return result


class TopoptEnergyModel(tf.keras.layers.Layer):
    def __init__(self, dim, dtype, exponent=3.0):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.exponent = self.add_weight("exponent", shape=[
        ], dtype=dtype, trainable=False, initializer=tf.keras.initializers.Constant(exponent))

    def set_exponent(self, val):
        self.set_weights([np.array(val)])

    def call(self, inputs):
        densities, dudx = inputs
        if self.dim == 2:
            energies = compute_lin_elasticity_strain_energy_2d(dudx)
            return tf.math.pow(densities, self.exponent) * energies
        if self.dim == 3:
            energies = compute_lin_elasticity_strain_energy_3d(dudx)
            return tf.math.pow(densities, self.exponent) * energies
        raise Exception('unsupported dim')


def compute_displacement_jacobian(disp_model, inputs, training=None):
    dim = disp_model.dim
    has_q = disp_model.has_q()
    if dim == 2:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            uxs = disp_model(inputs, training=training)
            uxs0 = tf.gather(uxs, 0, axis=1, batch_dims=0)
            uxs1 = tf.gather(uxs, 1, axis=1, batch_dims=0)
        if has_q:
            res00 = tf.gather(tape.gradient(uxs0, inputs), [0, 1], axis=1)
            res10 = tf.gather(tape.gradient(uxs1, inputs), [0, 1], axis=1)
        else:
            res00 = tape.gradient(uxs0, inputs)
            res10 = tape.gradient(uxs1, inputs)
        dudx = [res00, res10]
        del tape
        return dudx

    if dim == 3:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            uxs = disp_model(inputs, training=training)
            uxs0 = tf.gather(uxs, 0, axis=1, batch_dims=0)
            uxs1 = tf.gather(uxs, 1, axis=1, batch_dims=0)
            uxs2 = tf.gather(uxs, 2, axis=1, batch_dims=0)
        if has_q:
            res00 = tf.gather(tape.gradient(uxs0, inputs), [0, 1, 2], axis=1)
            res10 = tf.gather(tape.gradient(uxs1, inputs), [0, 1, 2], axis=1)
            res20 = tf.gather(tape.gradient(uxs2, inputs), [0, 1, 2], axis=1)
        else:
            res00 = tape.gradient(uxs0, inputs)
            res10 = tape.gradient(uxs1, inputs)
            res20 = tape.gradient(uxs2, inputs)
        dudx = [res00, res10, res20]
        del tape
        return dudx
    raise Exception('unsupported dim')


@tf.function
def compute_elasticity_energies(problem, disp_model, density_model, samples, training=None):
    energy_model = problem.get_energy_model()
    dudx = compute_displacement_jacobian(disp_model, samples, training=training)
    force_loss = problem.compute_force_loss(disp_model, samples)
    densities = density_model(samples, training=training)
    energy_densities = energy_model([densities, dudx])
    energy = problem.domain_volume * \
        tf.reduce_mean(energy_densities, keepdims=True)
    return energy, force_loss


@tf.custom_gradient
def compute_de_drho(densities, energy):
    def gradient_function(denergy):
        gradients = tf.gradients(energy, densities, grad_ys=denergy)[0]
        return -gradients, tf.zeros_like(energy)
    return energy, gradient_function


def compute_opt_energy(problem, disp_model, density_model, inputs, training=None):
    energy_model = problem.get_energy_model()

    dudx = compute_displacement_jacobian(disp_model, inputs, training=training)

    densities = density_model(inputs, training=training)

    energy_densities = energy_model([densities, dudx], training=training)

    energy = problem.domain_volume * \
        tf.reduce_mean(energy_densities, keepdims=True)
    # mock chain rule to be total derivative
    adjoint_e = compute_de_drho(densities, energy)

    return adjoint_e, densities


def compute_volume_penalty(densities, sample_volume, vol_penalty_strength, target_volume):
    volume_estimate = sample_volume * tf.reduce_mean(densities, keepdims=True)
    # divide by volume so penalty term behaves like a spring and penalty strength is like an actual material stiffness (~less dependent on the magnitude of target_volume)
    penalty_term = vol_penalty_strength * \
        (volume_estimate - target_volume) * \
        (volume_estimate - target_volume) / target_volume

    return penalty_term
