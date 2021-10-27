
import tensorflow as tf


def list_sum_size(py_list):
    return tf.math.reduce_sum([tf.size(q) for q in py_list])


def list_maximum(scalar_val, py_list):
    return [tf.math.maximum(scalar_val, q) for q in py_list]


def list_minimum(scalar_val, py_list):
    return [tf.math.minimum(scalar_val, q) for q in py_list]


def list_power(py_list, val):
    return [tf.math.pow(q, val) for q in py_list]


def list_add(py_list, move):
    return [(q + move) for q in py_list]


def list_divide(py_list, move):
    return [(q/move) for q in py_list]


def list_subtract(py_list, move):
    return [(q - move) for q in py_list]


def list_el_product(list1, list2):
    assert len(list1) == len(list2)
    return [list1[i] * list2[i] for i in range(len(list1))]


def list_el_minimum(list1, list2):
    assert len(list1) == len(list2)
    return [tf.math.minimum(list1[i], list2[i]) for i in range(len(list1))]


def list_el_maximum(list1, list2):
    assert len(list1) == len(list2)
    return [tf.math.maximum(list1[i], list2[i]) for i in range(len(list1))]


def compute_oc_multi_batch(old_densities, sensitivities, sample_volume, target_volume, max_move, damping_parameter):
    lagrange_lower_estimate = 0
    lagrange_upper_estimate = 1e9
    conv_threshold = 1e-3

    total_samples = list_sum_size(old_densities).numpy()
    dv = sample_volume / total_samples

    density_lower_bound = list_maximum(0.0, list_subtract(old_densities, max_move))
    density_upper_bound = list_minimum(1.0, list_add(old_densities, max_move))

    while (lagrange_upper_estimate-lagrange_lower_estimate)/(lagrange_lower_estimate+lagrange_upper_estimate) > conv_threshold:
        lagrange_current = 0.5 * \
            (lagrange_upper_estimate+lagrange_lower_estimate)

        target_densities = list_el_product(old_densities, list_power(
            list_divide(sensitivities, (- dv * lagrange_current)), damping_parameter))
        target_densities = list_el_maximum(
            density_lower_bound, list_el_minimum(density_upper_bound, target_densities))
        new_volume = sample_volume * \
            tf.math.reduce_mean([tf.math.reduce_mean(di)
                                for di in target_densities])

        if new_volume > target_volume:
            lagrange_lower_estimate = lagrange_current
        else:
            lagrange_upper_estimate = lagrange_current

    return target_densities
