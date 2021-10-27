
import tensorflow as tf


def pad_border(tensor, filter_size):
    rep = filter_size // 2
    tensor = tf.concat([tf.tile(tensor[:, :, 0:1, :], (1, 1, rep, 1)), tensor,  tf.tile(
        tensor[:, :, -1:, :], (1, 1, rep, 1))], axis=2)
    tensor = tf.concat([tf.tile(tensor[:, 0:1, :, :], (1, rep, 1,  1)), tensor, tf.tile(
        tensor[:, -1:, :, :], (1, rep, 1, 1))], axis=1)
    return tensor


def pad_border_3d(tensor, filter_size):
    rep = filter_size // 2
    tensor = tf.concat([tf.tile(tensor[:, :, :, 0:1, :], (1, 1, 1, rep, 1)), tensor,  tf.tile(
        tensor[:, :, :, -1:, :], (1, 1, 1, rep, 1))], axis=3)
    tensor = tf.concat([tf.tile(tensor[:, :, 0:1, :, :], (1, 1, rep, 1, 1)), tensor,  tf.tile(
        tensor[:, :, -1:, :, :], (1, 1, rep, 1,  1))], axis=2)
    tensor = tf.concat([tf.tile(tensor[:, 0:1, :, :, :], (1, rep, 1, 1,  1)), tensor, tf.tile(
        tensor[:, -1:, :, :, :], (1, rep, 1, 1,  1))], axis=1)
    return tensor


def pad_positions_constant(sample_positions, filter_size):
    rep = filter_size // 2
    #sample_positions = tf.concat([ tf.tile(sample_positions[:,:, 0:1,:], (1, 1, rep, 1) ), sample_positions,  tf.tile(sample_positions[:,:, -1:,:], (1, 1, rep, 1) ) ], axis = 2)
    c1 = tf.fill(tf.stack([tf.shape(sample_positions)[0], tf.shape(
        sample_positions)[1], rep, tf.shape(sample_positions)[3]]), -1000.0)
    sample_positions = tf.concat([c1, sample_positions, c1], axis=2)
    #sample_positions = tf.concat([ tf.tile(sample_positions[:, 0:1,:,:], (1, rep, 1,  1) ), sample_positions, tf.tile(sample_positions[:, -1:, :, :], (1, rep, 1, 1) ) ], axis = 1)
    c2 = tf.fill(tf.stack([tf.shape(sample_positions)[0], rep, tf.shape(
        sample_positions)[2], tf.shape(sample_positions)[3]]), -1000.0)
    sample_positions = tf.concat([c2, sample_positions, c2], axis=1)
    return sample_positions


def pad_positions_constant_3d(sample_positions, filter_size):
    rep = filter_size // 2
    c1 = tf.fill(tf.stack([tf.shape(sample_positions)[0], tf.shape(sample_positions)[
                 1], tf.shape(sample_positions)[2], rep, tf.shape(sample_positions)[4]]), -1000.0)
    sample_positions = tf.concat([c1, sample_positions, c1], axis=3)
    c2 = tf.fill(tf.stack([tf.shape(sample_positions)[0], tf.shape(sample_positions)[
                 1], rep, tf.shape(sample_positions)[3], tf.shape(sample_positions)[4]]), -1000.0)
    sample_positions = tf.concat([c2, sample_positions, c2], axis=2)
    c3 = tf.fill(tf.stack([tf.shape(sample_positions)[0], rep, tf.shape(sample_positions)[
                 2], tf.shape(sample_positions)[3], tf.shape(sample_positions)[4]]), -1000.0)
    sample_positions = tf.concat([c3, sample_positions, c3], axis=1)
    return sample_positions


@tf.function
def apply_sensitivity_filter_2d(sample_positions, old_densities, sensitivities, n_samples, domain, radius):
    dim = 2
    gamma = 1e-3

    cell_width = (domain[1] - domain[0]) / n_samples[0]
    grads = sensitivities

    radius_space = radius * cell_width
    filter_size = 2*round(radius) + 1
    density_patches = tf.reshape(
        old_densities, [1, n_samples[1], n_samples[0], 1])
    density_patches = pad_border(density_patches, filter_size)
    density_patches = tf.image.extract_patches(
        density_patches, sizes=[1, filter_size, filter_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')

    sensitivity_patches = tf.reshape(
        sensitivities, [1, n_samples[1], n_samples[0], 1])
    sensitivity_patches = pad_border(sensitivity_patches, filter_size)
    sensitivity_patches = tf.image.extract_patches(
        sensitivity_patches, sizes=[1, filter_size, filter_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')

    sample_positions = tf.reshape(
        sample_positions, [1, n_samples[1], n_samples[0], dim])
    # we pad such that influence is basically 0
    sample_patches = pad_positions_constant(
        sample_positions, filter_size)
    sample_patches = tf.image.extract_patches(
        sample_patches, sizes=[1, filter_size, filter_size, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
    # sample_patches.shape is now [1, rows, cols, filter_size ** 2 * * dim]

    diff = tf.reshape(sample_patches, [1, n_samples[1], n_samples[0], filter_size * filter_size,
                      dim]) - tf.reshape(sample_positions, [1, n_samples[1], n_samples[0], 1, dim])
    # [1, n_samples[1], n_samples[0], filter_size ** 2]
    dists = tf.math.sqrt(tf.math.reduce_sum(diff*diff, axis=4))

    # [1, n_samples[1], n_samples[0], filter_size ** 2]
    Hei = tf.math.maximum(0.0, radius_space - dists)
    # [1, n_samples[1], n_samples[0], filter_size ** 2]
    Heixic = Hei * density_patches * sensitivity_patches
    sum_Heixic = tf.math.reduce_sum(Heixic, axis=3)
    sum_Hei = tf.math.reduce_sum(Hei, axis=3)
    old_densities_r = tf.reshape(
        old_densities, [1, n_samples[1], n_samples[0]])
    assert len(sum_Hei.shape) == len(old_densities_r.shape)
    div = tf.math.maximum(gamma, old_densities_r) * sum_Hei
    grads = sum_Heixic / div

    grads = tf.reshape(grads, (-1, 1))
    return grads


@tf.function
def apply_sensitivity_filter_3d(sample_positions, old_densities, sensitivities, n_samples, domain, radius):
    dim = 3
    gamma = 1e-3

    cell_width = (domain[1] - domain[0]) / n_samples[0]
    radius_space = radius * cell_width
    filter_size = 2*round(radius) + 1
    sample_positions = tf.reshape(
        sample_positions, [1, n_samples[0], n_samples[1], n_samples[2], dim])
    density_patches = tf.reshape(
        old_densities, [1, n_samples[0], n_samples[1], n_samples[2], 1])
    density_patches = pad_border_3d(density_patches, filter_size)
    density_patches = tf.extract_volume_patches(
        density_patches, ksizes=[1, filter_size, filter_size, filter_size, 1], strides=[1, 1, 1, 1, 1], padding='VALID')

    sensitivity_patches = tf.reshape(
        sensitivities, [1, n_samples[0], n_samples[1], n_samples[2], 1])
    sensitivity_patches = pad_border_3d(sensitivity_patches, filter_size)
    sensitivity_patches = tf.extract_volume_patches(
        sensitivity_patches, ksizes=[1, filter_size, filter_size, filter_size, 1], strides=[1, 1, 1, 1, 1], padding='VALID')

    # we pad such that influence is basically 0
    sample_patches = pad_positions_constant_3d(sample_positions, filter_size)
    sample_patches = tf.extract_volume_patches(
        sample_patches, ksizes=[1, filter_size, filter_size, filter_size, 1], strides=[1, 1, 1, 1, 1], padding='VALID')
    # sample_patches.shape is now [1, rows, cols, filter_size ** 3 * * dim]

    diff = tf.reshape(sample_patches, [1, n_samples[0], n_samples[1], n_samples[2], filter_size * filter_size *
                      filter_size, dim]) - tf.reshape(sample_positions, [1, n_samples[0], n_samples[1], n_samples[2], 1, dim])
    # [1, n_samples[0], n_samples[1], n_samples[2], filter_size ** 3]
    dists = tf.math.sqrt(tf.math.reduce_sum(diff*diff, axis=5))

    # [1, n_samples[0], n_samples[1], n_samples[2], filter_size ** 3]
    Hei = tf.math.maximum(0.0, radius_space - dists)
    # [1, n_samples[0], n_samples[1], n_samples[2], filter_size ** 3]
    Heixic = Hei * density_patches * sensitivity_patches
    sum_Heixic = tf.math.reduce_sum(Heixic, axis=4)
    sum_Hei = tf.math.reduce_sum(Hei, axis=4)
    old_densities_r = tf.reshape(
        old_densities, [1, n_samples[0], n_samples[1], n_samples[2]])
    assert len(sum_Hei.shape) == len(old_densities_r.shape)
    div = tf.math.maximum(gamma, old_densities_r) * sum_Hei
    grads = sum_Heixic / div

    grads = tf.reshape(grads, (-1, 1))
    return grads


def apply_sensitivity_filter(sample_positions, old_densities, sensitivities, n_samples, domain, dim, radius):
    if dim == 2:
        return apply_sensitivity_filter_2d(sample_positions, old_densities, sensitivities, n_samples, domain, radius)
    if dim == 3:
        return apply_sensitivity_filter_3d(sample_positions, old_densities, sensitivities, n_samples, domain, radius)

    raise Exception('unsupported dim')
