
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ntopo.utils import get_grid_centers, compute_domain_volume


def zero_densities_function(positions):
    return tf.zeros(tf.stack((tf.shape(positions)[0], 1)), dtype=positions.dtype)


def one_densities_function(positions):
    return tf.ones(tf.stack((tf.shape(positions)[0], 1)), dtype=positions.dtype)


class DensityConstraintBase:

    def apply(self, positions, densities):
        distances, function_values = self.compute(positions)
        return tf.where(distances <= 0.0,  function_values, densities)

    def estimate_volume(self, domain):
        assert len(domain) == 4  # only implemented in 2d for now
        positions = get_grid_centers(domain, [512, 512], dtype=np.float32)
        zeros = tf.zeros((positions.shape[0], 1))
        ones = tf.ones((positions.shape[0], 1))
        distances, function_values = self.compute(positions)
        bounding_volume = compute_domain_volume(domain)
        free_volume = bounding_volume * \
            tf.math.reduce_mean(tf.where(distances >= 0.0,  ones, zeros))
        constrained_densities = tf.where(distances < 0.0, function_values, zeros)
        constraint_volume = bounding_volume * tf.math.reduce_mean(constrained_densities)
        return free_volume.numpy(), constraint_volume.numpy()

    def plot(self, domain, n_samples, folder):
        positions = get_grid_centers(domain, n_samples)
        distances, function_values = self.compute(positions)

        distances = np.reshape(distances, (n_samples[1], n_samples[0]))
        plt.figure()
        plt.imshow(np.flipud(distances), extent=domain)
        plt.colorbar()
        plt.savefig(os.path.join(folder, 'density_constraint_phi.png'))
        plt.close()

        function_values = np.reshape(function_values, (n_samples[1], n_samples[0]))
        plt.figure()
        plt.imshow(np.flipud(function_values), extent=domain)
        plt.colorbar()
        plt.savefig(os.path.join(folder, 'density_constraint_fs.png'))
        plt.close()

    def compute(self, positions):
        raise Exception("compute not overriden")


class DensityConstraint(DensityConstraintBase):
    def __init__(self, sdf, fun):
        super().__init__()
        self.sdf = sdf
        self.fun = fun

    def compute(self, positions):
        distances = self.sdf.eval_distance(positions)
        function_values = self.fun(positions)
        return distances, function_values

    def has_constraint(self):
        return True

    def plot_boundary(self, *args, **kwargs):
        self.sdf.plot_boundary(*args, **kwargs)


class DensityConstraintNone(DensityConstraintBase):
    def apply(self, positions, densities):
        return densities

    def plot(self, domain, n_samples, folder):
        pass

    def has_constraint(self):
        return False

    def plot_boundary(self, *args, **kwargs):
        pass


class DensityConstraintAdd(DensityConstraintBase):
    def __init__(self, constraint_list):
        super().__init__()
        self.constraint_list = constraint_list

    def compute(self, positions):
        distances, function_values = self.constraint_list[0].compute(positions)
        for i in range(1, len(self.constraint_list)):
            phin, fxsn = self.constraint_list[i].compute(positions)
            cond = phin < 0.0
            distances = tf.where(phin < distances, phin, distances)
            function_values = tf.where(cond, fxsn, function_values)
        return distances, function_values

    def has_constraint(self):
        for constraint_object in self.constraint_list:
            if constraint_object.has_constraint():
                return True
        return False

    def plot_boundary(self, *args, **kwargs):
        for constraint_object in self.constraint_list:
            constraint_object.plot_boundary(*args, **kwargs)


class DisplacementLine:
    def __init__(self, point_a, point_b):
        super().__init__()
        self.point_a = np.reshape(np.array(point_a, dtype=np.float32), (1, 2))
        self.point_b = np.reshape(np.array(point_b, dtype=np.float32), (1, 2))
        self.ba_sq = np.sum((self.point_b - self.point_a) * (self.point_b - self.point_a))

    def eval_distance_square(self, positions):
        a_to_p = positions - self.point_a
        ba = self.point_b - self.point_a
        line_time = tf.clip_by_value(tf.math.reduce_sum(
            a_to_p*ba, axis=1, keepdims=True)/self.ba_sq, 0.0, 1.0)
        d = a_to_p - line_time * ba
        dist_sq = tf.math.reduce_sum(d*d, axis=1, keepdims=True)
        return dist_sq


class DisplacementDisk:
    def __init__(self, center, radius):
        super().__init__()
        self.center = tf.convert_to_tensor(np.reshape(
            np.array(center, dtype=np.float32), (1, 2)), dtype=tf.float32)
        self.radius = radius

    def eval_distance_square(self, positions):
        center_to_p = positions - self.center
        dist_sq_from_center = tf.math.reduce_sum(center_to_p*center_to_p, axis=1, keepdims=True)
        d = tf.math.sqrt(dist_sq_from_center)
        dist_sq = tf.where(d - self.radius <= 1e-5,
                           tf.zeros_like(d), tf.math.square(d - self.radius))
        return dist_sq


class DisplacementPoint:
    def __init__(self, position):
        super().__init__()
        self.position = tf.convert_to_tensor(np.reshape(
            np.array(position, dtype=np.float32), (1, 2)), dtype=tf.float32)

    def eval_distance_square(self, positions):
        p_to_positions = positions - self.position
        dist_sq_from_center = tf.math.reduce_sum(p_to_positions*p_to_positions, axis=1, keepdims=True)
        return dist_sq_from_center


class DisplacementHalfspace:
    def __init__(self, normal, offset):
        super().__init__()
        self.normal = tf.convert_to_tensor(np.reshape(
            np.array(normal, dtype=np.float32), (2, 1)), dtype=tf.float32)
        self.offset = offset

    def eval_distance_square(self, positions):
        signed_distances = tf.linalg.matmul(positions, self.normal) + self.offset
        return signed_distances*signed_distances


def power_smooth_min_already_sq(d_a, d_b):
    eps = 1e-4
    return (d_a*d_b)/(d_a+d_b + eps)


def tree_reduce(d_sq):
    while len(d_sq) > 1:
        nd_sq = []
        for i in range(0, len(d_sq), 2):
            if i + 1 < len(d_sq):
                nd_sq.append(power_smooth_min_already_sq(d_sq[i], d_sq[i+1]))
            else:
                nd_sq.append(d_sq[i])
        d_sq = nd_sq
    return d_sq[0]


def entity_list_compute_s(positions, domain, entity_list):
    assert len(positions.shape) == 2
    d_sq = []
    for entity_i in entity_list:
        d_sqi = entity_i.eval_distance_square(positions)
        d_sq.append(d_sqi)
    d_sq = tree_reduce(d_sq)
    result = 2.0 / max(domain[1] - domain[0], domain[3] -
                       domain[2]) * tf.math.sqrt(d_sq + 1e-35)
    return result


def get_gradient_norm_function(functor):
    @tf.function
    def gradient_norm_function(inputs):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            T = functor(inputs)
        dTdxy = tape.gradient(T, inputs)
        return tf.math.sqrt(tf.math.reduce_sum(tf.square(dTdxy), axis=1, keepdims=True))
    return gradient_norm_function


def get_second_order_norm_function(functor):
    @tf.function
    def norm_function(inputs):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(inputs)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(inputs)
                T = functor(inputs)
            dTdxy = tape.gradient(T, inputs, unconnected_gradients='zero')
            dTdx = tf.gather(dTdxy, [0], axis=1)
            dTdy = tf.gather(dTdxy, [1], axis=1)

        dTdxdxy = tape1.gradient(dTdx, inputs, unconnected_gradients='zero')
        dTdydxy = tape1.gradient(dTdy, inputs, unconnected_gradients='zero')

        return tf.math.sqrt(tf.math.reduce_sum(tf.square(dTdxdxy) + tf.square(dTdydxy), axis=1, keepdims=True))
    return norm_function


class DisplacementConstraint:
    def __init__(self, domain, entity_list):
        self.domain = domain
        self.entity_list = entity_list

    def compute_length_factor(self, positions):
        return entity_list_compute_s(positions, self.domain, self.entity_list)

    def plot(self, domain, n_samples, folder):
        positions = get_grid_centers(domain, n_samples, dtype=np.float32)
        length_factors = self.compute_length_factor(positions)

        length_factors = np.reshape(length_factors, (n_samples[1], n_samples[0]))
        plt.figure()
        plt.imshow(np.flipud(length_factors), extent=domain)
        plt.colorbar(orientation="horizontal")
        plt.savefig(os.path.join(folder, 'DisplacementConstraint_s.png'))
        plt.close()

        s_function = self.compute_length_factor

        gradient_norm_function = get_gradient_norm_function(s_function)
        samples_tf = tf.convert_to_tensor(positions)
        gradient_norms = gradient_norm_function(samples_tf)
        image = np.reshape(gradient_norms, (n_samples[1], n_samples[0]))
        plt.figure()
        plt.imshow(np.flipud(image), extent=self.domain)
        plt.colorbar(orientation="horizontal")
        plt.savefig(os.path.join(folder, 'DisplacementConstraint_s-gradient-norm.png'))
        plt.close()

        so_function = get_second_order_norm_function(s_function)
        samples_tf = tf.convert_to_tensor(positions)
        so_norms = so_function(samples_tf)
        image = np.reshape(so_norms, (n_samples[1], n_samples[0]))
        plt.figure()
        plt.imshow(np.flipud(image), extent=self.domain)
        plt.colorbar(orientation="horizontal")
        plt.savefig(os.path.join(
            folder, 'DisplacementConstraint_s-so-norm.png'))
        plt.close()
