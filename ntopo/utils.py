
import math
import json
import random as rn
import numpy as np
import tensorflow as tf
import itertools

def set_random_seed(seed):
    np.random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)


def tf_to_np_data_type(dtype):
    if dtype == tf.float32:
        return np.float32
    if dtype == tf.float64:
        return np.float64

    raise ValueError("tf_to_np_data_type: not supported data type")


def compute_domain_volume(domain):
    """
    computes volume but if length[i] is zero, it is ignored
    """
    domain = np.array(domain)
    lower = domain[0:len(domain):2]
    upper = domain[1:len(domain):2]
    length = upper - lower
    length[length == 0.0] = 1.0
    return np.prod(length)


def scale_domain(domain, scale_factor):
    width = domain[1] - domain[0]
    height = domain[3] - domain[2]
    eps = scale_factor - 1.0
    x0 = domain[0] - width * 0.5 * eps
    x1 = domain[1] + width * 0.5 * eps
    y0 = domain[2] - height * 0.5 * eps
    y1 = domain[3] + height * 0.5 * eps
    return [x0, x1, y0, y1]


def get_grid_points(domain, n_cells, dtype=np.float32):
    """
    in 2D uses 'xy' indexing and in 3D 'ij' indexing
    returns a tensor with shape (n, dim)
    """
    domain = np.array(domain, dtype=dtype)
    n_cells = np.array(n_cells)
    assert n_cells.size in (2, 3)
    n = np.prod(n_cells)

    if n_cells.size == 2:
        x = np.linspace(domain[0], domain[1], num=n_cells[0], dtype=dtype)
        y = np.linspace(domain[2], domain[3], num=n_cells[1], dtype=dtype)
        xs, ys = np.meshgrid(x, y)

        xs = np.reshape(xs, (n, 1))
        ys = np.reshape(ys, (n, 1))
        result = np.hstack((xs, ys))
        return result
    if n_cells.size == 3:
        x = np.linspace(domain[0], domain[1], num=n_cells[0], dtype=dtype)
        y = np.linspace(domain[2], domain[3], num=n_cells[1], dtype=dtype)
        z = np.linspace(domain[4], domain[5], num=n_cells[2], dtype=dtype)
        xs, ys, zs = np.meshgrid(x, y, z, indexing='ij')

        xs = np.reshape(xs, (n, 1))
        ys = np.reshape(ys, (n, 1))
        zs = np.reshape(zs, (n, 1))
        result = np.hstack((xs, ys, zs))
        return result
    raise Exception('not supported')

def get_grid_centers_spacing(domain, n_cells, dtype=np.float32):
    domain = np.array(domain, dtype=dtype)
    n_cells = np.array(n_cells, dtype=np.int32)
    assert n_cells.size in (2, 3)
    width = domain[1] - domain[0]
    height = domain[3] - domain[2]
    if n_cells.size == 3:
        length = domain[5] - domain[4]
        cx = 0.5 * width/n_cells[0]
        cy = 0.5 * height/n_cells[1]
        cz = 0.5 * length/n_cells[2]
        return [cx, cy, cz]
    raise Exception('not supported')

def get_grid_centers(domain, n_cells, dtype=np.float32):
    """
    in 2D uses 'xy' indexing and in 3D 'ij' indexing
    returns a tensor with shape (n, dim)
    """
    domain = np.array(domain, dtype=dtype)
    n_cells = np.array(n_cells, dtype=np.int32)
    assert n_cells.size in (2, 3)
    n = np.prod(n_cells)
    width = domain[1] - domain[0]
    height = domain[3] - domain[2]
    if n_cells.size == 2:
        x = np.linspace(domain[0] + 0.5 * width/n_cells[0], domain[1] -
                        0.5 * width/n_cells[0], num=n_cells[0], dtype=dtype)
        y = np.linspace(domain[2] + 0.5 * height/n_cells[1], domain[3] -
                        0.5 * height/n_cells[1], num=n_cells[1], dtype=dtype)
        xs, ys = np.meshgrid(x, y)

        xs = np.reshape(xs, (n, 1))
        ys = np.reshape(ys, (n, 1))
        result = np.hstack((xs, ys))

    else:
        length = domain[5] - domain[4]
        cx = 0.5 * width/n_cells[0]
        cy = 0.5 * height/n_cells[1]
        cz = 0.5 * length/n_cells[2]
        x = np.linspace(domain[0] + cx, domain[1] - cx, num=n_cells[0], dtype=dtype)
        y = np.linspace(domain[2] + cy, domain[3] - cy, num=n_cells[1], dtype=dtype)
        z = np.linspace(domain[4] + cz, domain[5] - cz, num=n_cells[2], dtype=dtype)
        xs, ys, zs = np.meshgrid(x, y, z, indexing='ij')
        xs = np.reshape(xs, (n, 1))
        ys = np.reshape(ys, (n, 1))
        zs = np.reshape(zs, (n, 1))
        result = np.hstack((xs, ys, zs))
    return result

def stratified_sampling(domain, n_cells, n_points_per_cell=1, dtype=np.float32):
    assert isinstance(domain, (list, np.ndarray)
                      ), "domain has to be python list or numpy array"
    assert isinstance(n_cells, (list, np.ndarray)
                      ), "n_cells has to be python list or numpy array"
    assert np.array(domain).size == 2 * np.array(n_cells).size

    nx = n_cells[0]

    left = domain[0]
    right = domain[1]
    if len(n_cells) == 1:
        n = nx * n_points_per_cell

        lefts = np.linspace(left, right, nx, endpoint=False)
        lefts = np.reshape(lefts, (nx, 1))
        rights = lefts + (right - left) / nx
        ntile = n_points_per_cell
        lefts = np.tile(lefts, (ntile, 1))
        rights = np.tile(rights, (ntile, 1))

        l1 = np.random.uniform(0.0, 1.0, (n, 1))
        xs = (1.0 - l1) * lefts + l1 * rights
        return xs.astype(dtype)

    if len(n_cells) == 2:
        bottom = domain[2]
        top = domain[3]
        ny = n_cells[1]
        n = nx * ny*n_points_per_cell

        leftsl = np.linspace(left, right, nx, endpoint=False)
        rightsl = leftsl + (right - left) / nx
        bottomsl = np.linspace(bottom, top, ny, endpoint=False)
        topsl = bottomsl + (top - bottom) / ny
        leftsv, bottomsv = np.meshgrid(leftsl, bottomsl)
        rightsv, topsv = np.meshgrid(rightsl, topsl)

        lefts = np.reshape(leftsv, (nx * ny, 1))
        rights = np.reshape(rightsv, (nx * ny, 1))
        bottoms = np.reshape(bottomsv, (nx * ny, 1))
        tops = np.reshape(topsv, (nx * ny, 1))

        ntile = n_points_per_cell
        lefts = np.tile(lefts, (ntile, 1))
        rights = np.tile(rights, (ntile, 1))
        bottoms = np.tile(bottoms, (ntile, 1))
        tops = np.tile(tops, (ntile, 1))

        l1 = np.random.uniform(0.0, 1.0, (n, 1))
        l2 = np.random.uniform(0.0, 1.0, (n, 1))
        xs = (1.0 - l1) * lefts + l1 * rights
        ys = (1.0 - l2) * bottoms + l2 * tops
        result = np.hstack((xs, ys))
        return result.astype(dtype)

    if len(n_cells) == 3:
        bottom = domain[2]
        top = domain[3]
        ny = n_cells[1]
        nz = n_cells[2]
        near = domain[4]
        far = domain[5]

        n = nx * ny * nz * n_points_per_cell

        leftsl = np.linspace(left, right, nx, endpoint=False)
        rightsl = leftsl + (right - left) / nx
        bottomsl = np.linspace(bottom, top, ny, endpoint=False)
        topsl = bottomsl + (top - bottom) / ny
        nearsl = np.linspace(near, far, nz, endpoint=False)
        farsl = nearsl + (far - near) / nz

        leftsv, bottomsv, nearsv = np.meshgrid(
            leftsl, bottomsl, nearsl, indexing='ij')
        rightsv, topsv, farsv = np.meshgrid(
            rightsl, topsl, farsl, indexing='ij')

        lefts = np.reshape(leftsv, (nx * ny * nz, 1))
        rights = np.reshape(rightsv, (nx * ny * nz, 1))
        bottoms = np.reshape(bottomsv, (nx * ny * nz, 1))
        tops = np.reshape(topsv, (nx * ny * nz, 1))
        fars = np.reshape(farsv, (nx * ny * nz, 1))
        nears = np.reshape(nearsv, (nx * ny * nz, 1))

        ntile = n_points_per_cell
        lefts = np.tile(lefts, (ntile, 1))
        rights = np.tile(rights, (ntile, 1))
        bottoms = np.tile(bottoms, (ntile, 1))
        tops = np.tile(tops, (ntile, 1))
        nears = np.tile(nears, (ntile, 1))
        fars = np.tile(fars, (ntile, 1))

        l1 = np.random.uniform(0.0, 1.0, (n, 1))
        l2 = np.random.uniform(0.0, 1.0, (n, 1))
        l3 = np.random.uniform(0.0, 1.0, (n, 1))
        xs = (1.0 - l1) * lefts + l1 * rights
        ys = (1.0 - l2) * bottoms + l2 * tops
        zs = (1.0 - l3) * nears + l3 * fars
        result = np.hstack((xs, ys, zs))
        return result.astype(dtype)

    raise Exception('unsupported ncells')

def get_sample_generator(domain, n_samples):
    def _get_sample_generator(domain, n_samples):
        while True:
            xs = stratified_sampling(domain, n_samples)
            xs = tf.convert_to_tensor(xs, dtype=tf.float32)
            yield xs
    dim = len(domain)//2
    dataset = tf.data.Dataset.from_generator(lambda : _get_sample_generator(domain, n_samples), tf.float32, tf.TensorShape([np.prod(n_samples), dim]) )
    dataset = dataset.prefetch(1)
    iterator = dataset.__iter__()
    return iterator

def get_single_random_q_sample_generator(q_domain, domain, n_samples):
    def _get_sample_generator(domain, n_samples):
        while True:
            q = stratified_sampling(
                q_domain, [ 1 ], n_points_per_cell=1, dtype=np.float32)
            q_vec = np.ones((np.prod(n_samples), 1), dtype=np.float32) * q
            xs = stratified_sampling(
                domain, n_samples, n_points_per_cell=1, dtype=np.float32)
            xs = np.concatenate((xs, q_vec), axis=1)
            xs = tf.convert_to_tensor(xs, dtype=tf.float32)
            yield xs
    total_dim = len(domain)//2 + len(q_domain) // 2
    dataset = tf.data.Dataset.from_generator(lambda : _get_sample_generator(domain, n_samples), tf.float32, tf.TensorShape([np.prod(n_samples), total_dim]) )
    dataset = dataset.prefetch(1)
    iterator = dataset.__iter__()
    return iterator

def get_q_sample_generator(q, domain, n_samples):
    q_vec = np.ones((np.prod(n_samples), 1), dtype=np.float32) * q
    def _get_sample_generator(domain, n_samples):
        while True:
            xs = stratified_sampling(
                domain, n_samples, n_points_per_cell=1, dtype=np.float32)
            xs = np.concatenate((xs, q_vec), axis=1)
            xs = tf.convert_to_tensor(xs, dtype=tf.float32)
            yield xs
    total_dim = len(domain)//2 + q_vec.shape[1]
    dataset = tf.data.Dataset.from_generator(lambda : _get_sample_generator(domain, n_samples), tf.float32, tf.TensorShape([np.prod(n_samples), total_dim]) )
    dataset = dataset.prefetch(1)
    iterator = dataset.__iter__()
    return iterator


def get_default_figure_size(domain):
    width = domain[1] - domain[0]
    height = domain[3] - domain[2]
    if width < height:
        return (round(12 * width / height), 12)

    return (12, round(12 * height / width))


def get_default_sample_counts(domain, total_number_of_samples, even=None):
    assert len(domain) == 4 or len(domain) == 6
    if len(domain) == 4:
        width = domain[1] - domain[0]
        height = domain[3] - domain[2]
        def relative_error(nx_ny):
            nx, ny = nx_ny
            aspect_ratio_n = nx/ny
            aspect_ratio = width/height
            return abs(aspect_ratio - aspect_ratio_n) / aspect_ratio
        def best_possibility(possibilities):
            if even:
                possibilities[0] = [n for n in possibilities[0] if n % 2 == 0] # filter out odd numbers
                possibilities[1] = [n for n in possibilities[1] if n % 2 == 0]
            # compute all possibile combinations of candidate nx and ny
            p_nx_ny = list(itertools.product(possibilities[0], possibilities[1]))
            errors = map(relative_error, p_nx_ny)
            min_index, _ = min(enumerate(errors), key=lambda x: x[1])
            return list(p_nx_ny[min_index])
        possibilities = [[], []]
        # we try to match the aspect ratio of (width, height) with (nx, ny) i.e. width/height ~= nx / ny and nx * ny ~= total_number_of_samples
        #  (ny * width / height) * ny ~= total_number_of_samples, ny ~= sqrt( total_number_of_samples * height / width )
        ny = int(math.floor(math.sqrt(total_number_of_samples * height / width)))
        possibilities[1].append(ny)
        possibilities[1].append(ny + 1)
        
        nx = int(math.floor(ny * width / height))
        possibilities[0].append(nx)
        possibilities[0].append(nx + 1)
        
        #now lets try opposite order
        # nx * nx /total_number_of_samples  ~= width/height, nx ~= sqrt( total_number_of_samples * width / height )
        nx = int(math.floor(math.sqrt(total_number_of_samples * width / height)))
        possibilities[0].append(nx)
        possibilities[0].append(nx + 1)
        ny = int(math.floor(nx * height / width))
        possibilities[1].append(ny)
        possibilities[1].append(ny + 1)
        return best_possibility(possibilities)
        
    if len(domain) == 6:
        if even:
            raise Exception('unsupported ')

        width = domain[1] - domain[0]
        height = domain[3] - domain[2]
        depth = domain[5] - domain[4]
        # we try to match the aspect ratio of (width, height, depth) with (nx, ny, nz) i.e. nx * ny * nz ~= total_number_of_samples  and  nx ~= scale * width, ny ~= scale * height, nz ~= scale * depth

        scale = pow(total_number_of_samples / (width * height * depth), 1.0 / 3.0)
        nx = int(round(scale * width))
        ny = int(round(nx * height / width))
        nz = int(round(nx * depth / width))

        return [nx, ny, nz]
    raise Exception('unsupported domain')


def transform_minus11(xs, domain):
    dim = domain.size // 2
    dn = len(xs.shape)
    shape_v = np.ones((dn, ), dtype=np.int32)
    # construct shape vector such that ones except last one which is DIM, which allows for multiple batch dimensions
    shape_v[-1] = dim

    lower_indices = list(range(0, domain.size, 2))
    upper_indices = list(range(1, domain.size, 2))
    starts = np.reshape(domain[lower_indices], shape_v)
    widths = np.reshape(domain[upper_indices] - domain[lower_indices], shape_v)

    return 2.0 / widths * xs + (- 1.0 - 2.0 * starts / widths)


def write_to_file(some_string, filename):
    with open(filename, 'w') as file_object:
        file_object.write(some_string)


def read_from_json_file(filename):
    with open(filename) as file_object:
        data = json.load(file_object)
    return data


def load_file(filename):
    with open(filename, 'r') as file_object:
        contents = file_object.read()
    return contents


def write_to_json_file(obj, filename):
    with open(filename, 'w') as file_object:
        json.dump(obj, file_object, indent=2)
