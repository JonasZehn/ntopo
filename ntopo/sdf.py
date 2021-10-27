import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ntopo.utils import get_grid_centers


class SDF:
    """
    A class that represents a signed distance function, where the output is negative for inside, and positive for outside
    """

    def __init__(self):
        pass

    def eval_distance(self, positions):
        raise Exception('not overriden method')

    def plot(self, domain, n_samples, folder):
        positions = get_grid_centers(domain, n_samples)
        dists = self.eval_distance(positions)
        dists = np.reshape(dists, (n_samples[1], n_samples[0]))
        plt.figure()
        plt.imshow(np.flipud(dists), extent=domain)
        plt.colorbar()
        plt.savefig(os.path.join(folder, 'sdf.png'))
        plt.close()

        plt.figure()
        plt.imshow(np.flipud(np.where(dists < 0.0, np.ones_like(dists),
                                      np.zeros_like(dists))), extent=domain)
        plt.colorbar()
        plt.savefig(os.path.join(folder, 'sdf-smaller-than-zero.png'))
        plt.close()


class SDFDisk(SDF):
    def __init__(self, center, radius, inside=True):
        super().__init__()
        self.center = center.copy()
        self.center_tf = tf.convert_to_tensor(np.reshape(
            np.array(center, dtype=np.float32), (1, 2)), dtype=tf.float32)
        self.radius = radius
        self.inside = inside

    def eval_distance(self, positions):
        center_to_p = positions - self.center_tf
        dist_sq_from_center = tf.math.reduce_sum(center_to_p*center_to_p, axis=1, keepdims=True)
        distance = tf.math.sqrt(dist_sq_from_center)
        #rs = self.center + self.radius * center_to_p/distance
        #color = self.colorfun(rs)
        if self.inside:
            return distance - self.radius

        return self.radius - distance

    def plot_boundary(self, axes, color, linewidth):
        circle = plt.Circle(self.center, self.radius, color=color,
                            linewidth=linewidth, fill=False)
        axes.add_patch(circle)


class SDFHalfSpace(SDF):
    """
    A half spacce defined by a normal and an offset ; x.dot(normal) + offset <= 0
    """

    def __init__(self, normal, offset):
        super().__init__()
        self.normal = np.reshape(np.array(normal, dtype=np.float32), (2, 1))
        self.offset = offset

    def eval_distance(self, positions):
        assert len(positions.shape) == 2
        signed_distance = tf.linalg.matmul(positions, self.normal) + self.offset
        assert len(signed_distance.shape) == 2
        return signed_distance


class SDFUnion(SDF):
    """
    definition  phi < 0 inside, => union, any phi_i < 0
    """

    def __init__(self, sdf_list):
        super().__init__()
        self.sdf_list = sdf_list

    def eval_distance(self, positions):
        distances = []
        for child in self.sdf_list:
            child_dist = child.eval_distance(positions)
            assert len(child_dist.shape) == 2
            distances.append(child_dist)
        all_dists = tf.concat(distances, axis=1)
        return tf.math.reduce_min(all_dists, axis=1, keepdims=True)

    def plot_boundary(self, axes, color, linewidth):
        raise Exception('not implemented method')


class SDFIntersection(SDF):
    def __init__(self, sdf_list):
        super().__init__()
        self.sdf_list = sdf_list

    def eval_distance(self, positions):
        distances = []
        for child in self.sdf_list:
            child_dist = child.eval_distance(positions)
            assert len(child_dist.shape) == 2
            distances.append(child_dist)
        all_dists = tf.concat(distances, axis=1)
        return tf.math.reduce_max(all_dists, axis=1, keepdims=True)


class SDFRectangle(SDF):
    def __init__(self, rectangle):
        super().__init__()
        self.rectangle = rectangle.copy()
        sdf_list = [
            SDFHalfSpace([-1.0, 0.0], rectangle[0]),
            SDFHalfSpace([1.0, 0.0], -rectangle[1]),
            SDFHalfSpace([0.0, -1.0], rectangle[2]),
            SDFHalfSpace([0.0, 1.0], -rectangle[3]),
        ]
        self.sub = SDFIntersection(sdf_list)

    def eval_distance(self, positions):
        return self.sub.eval_distance(positions)

    def width(self):
        return self.rectangle[1] - self.rectangle[0]

    def height(self):
        return self.rectangle[3] - self.rectangle[2]

    def plot_boundary(self, axes, color, linewidth):
        rect = plt.Rectangle((self.rectangle[0], self.rectangle[2]), self.width(),
                             self.height(), color=color, linewidth=linewidth, fill=False)
        axes.add_patch(rect)
