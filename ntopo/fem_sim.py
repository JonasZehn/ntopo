"""
A simple 2D FEM simulation using quadrilaterals
to show the compliance computation using FEM.
Following the convention of the other code, indexing is xy/cartesian
and not ij (see doc. of numpy.meshgrid)
"""

import os
import sys
import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from numba import jit, njit
import matplotlib.pyplot as plt
from ntopo.utils import write_to_file

import cvxopt
from cvxopt import cholmod

@njit("float32[:,:](float32, float32, float32)")
def element_stiffness_over_youngs_modulus(nu, w, h):
    """
    computes stiffness for a quadrilateral with bilinear interpolation basis function, with width w and height h
    K is linear in youngs modulus, so we return K/E for the point of topopt
    variable order
    u[1, 0]  -  u[1, 1]
    h |           |
    u[0, 0]  -  u[0, 1]
             w
    dof_order = [u[0,0], u[0, 1], u[1, 0], u[1, 1]]
    generated code using sympy 
    """
    tmp0 = 1.0/h
    tmp1 = tmp0*w
    tmp2 = (1.0/6.0)*tmp1
    tmp3 = 1.0/w
    tmp4 = h*tmp3
    tmp5 = (1.0/3.0)*tmp4
    tmp6 = -nu*tmp2 + tmp2 + tmp5
    tmp7 = (1.0/8.0)*nu
    tmp8 = tmp7 + 1.0/8.0
    tmp9 = (1.0/12.0)*tmp1
    tmp10 = -nu*tmp9 - tmp5 + tmp9
    tmp11 = (3.0/8.0)*nu
    tmp12 = tmp11 - 1.0/8.0
    tmp13 = pow(h, 2)
    tmp14 = pow(w, 2)
    tmp15 = nu - 1
    tmp16 = tmp14*tmp15
    tmp17 = tmp0*tmp3
    tmp18 = (1.0/6.0)*tmp17
    tmp19 = tmp18*(tmp13 + tmp16)
    tmp20 = 1.0/8.0 - tmp11
    tmp21 = 2*tmp13
    tmp22 = (1.0/12.0)*tmp17
    tmp23 = tmp22*(tmp16 - tmp21)
    tmp24 = -tmp7 - 1.0/8.0
    tmp25 = (1.0/3.0)*tmp1
    tmp26 = (1.0/6.0)*tmp4
    tmp27 = -nu*tmp26 + tmp25 + tmp26
    tmp28 = tmp18*(nu*tmp13 - tmp13 + tmp14)
    tmp29 = (1.0/12.0)*tmp4
    tmp30 = -nu*tmp29 - tmp25 + tmp29
    tmp31 = 2*tmp14
    tmp32 = tmp13*tmp15
    tmp33 = tmp22*(-tmp31 + tmp32)
    tmp34 = 1 - nu
    tmp35 = tmp14*tmp34
    tmp36 = tmp18*(tmp21 + tmp35)
    tmp37 = tmp22*(-4*tmp13 + tmp35)
    tmp38 = tmp18*(tmp13*tmp34 + tmp31)
    tmp39 = tmp18*(tmp14 + tmp32)
    mat = np.array([
        [ tmp6,  tmp8, tmp10, tmp12, tmp19, tmp20, tmp23, tmp24],
        [ tmp8, tmp27, tmp20, tmp28, tmp12, tmp30, tmp24, tmp33],
        [tmp10, tmp20,  tmp6, tmp24, tmp23,  tmp8, tmp19, tmp12],
        [tmp12, tmp28, tmp24, tmp27,  tmp8, tmp33, tmp20, tmp30],
        [tmp19, tmp12, tmp23,  tmp8, tmp36, tmp24, tmp37, tmp20],
        [tmp20, tmp30,  tmp8, tmp33, tmp24, tmp38, tmp12, tmp39],
        [tmp23, tmp24, tmp19, tmp20, tmp37, tmp12, tmp36,  tmp8],
        [tmp24, tmp33, tmp12, tmp30, tmp20, tmp39,  tmp8, tmp38]], dtype=np.float32)
    return np.float32(1.0/(1-nu**2))*mat

def compute_element_idcs(n_vertices_x, n_vertices_y):
    n_elements = (n_vertices_x - 1 ) * (n_vertices_y - 1)
    elements = np.zeros((n_elements, 4), dtype=np.int32)
    for element_i in range(n_vertices_y - 1):
        for element_j in range(n_vertices_x - 1):
            element_idx = element_i  * (n_vertices_x - 1) + element_j
            element = [
                element_i * n_vertices_x + element_j,
                element_i * n_vertices_x + (element_j + 1),
                (element_i + 1) * n_vertices_x + element_j,
                (element_i + 1) * n_vertices_x + (element_j + 1)
            ]
            elements[element_idx, :] = element
    return elements

@njit
def compute_stiffness_matrix(element_width, element_height,
    n_vertices_x, n_vertices_y, fixed, element_youngs_moduli,
    dat, row, col):
    nu = 0.3
    ki = element_stiffness_over_youngs_modulus(nu, element_width, element_height)
    current_idx = 0
    for i in range(n_vertices_y):
        for j in range(n_vertices_x):
            vtx_idx = i * n_vertices_x + j
            if fixed[2 * vtx_idx + 0]:
                dat[current_idx] = 1.0
                row[current_idx] = 2 * vtx_idx
                col[current_idx] = 2 * vtx_idx
                current_idx += 1
            if fixed[2 * vtx_idx + 1]:
                dat[current_idx] = 1.0
                row[current_idx] = 2 * vtx_idx + 1
                col[current_idx] = 2 * vtx_idx + 1
                current_idx += 1

    for element_i in range(n_vertices_y - 1):
        for element_j in range(n_vertices_x - 1):
            element_idx = element_i  * (n_vertices_x - 1) + element_j
            element = [
                element_i * n_vertices_x + element_j,
                element_i * n_vertices_x + (element_j + 1),
                (element_i + 1) * n_vertices_x + element_j,
                (element_i + 1) * n_vertices_x + (element_j + 1)
            ]
            idcs = [2 * element[0], 2 * element[0] + 1,
                   2 * element[1], 2 * element[1] + 1,
                   2 * element[2], 2 * element[2] + 1,
                   2 * element[3], 2 * element[3] + 1]

            fixed_i = []
            for j in range(8):
                fixed_i.append(fixed[idcs[j]])

            element_youngs_modulus = element_youngs_moduli[element_idx]
            for j in range(8):
                for k in range(8):
                    if (not fixed_i[j]) and (not fixed_i[k]):
                        dat[current_idx] = element_youngs_modulus * ki[j, k]
                        row[current_idx] = idcs[j]
                        col[current_idx] = idcs[k]
                        current_idx += 1

    return current_idx


def compute_x_undef(n_vertices_x, n_vertices_y, element_width, element_height):
    x_undef = np.zeros((n_vertices_y, n_vertices_x, 2))
    for i in range(n_vertices_y):
        for j in range(n_vertices_x):
            x_undef[i, j, 0] = j * element_width
            x_undef[i, j, 1] = i * element_height
    return x_undef

def compute_element_centers(n_vertices_x, n_vertices_y, x):
    element_centers = np.zeros((n_vertices_y - 1, n_vertices_x - 1, 2))
    for i in range(n_vertices_y - 1):
        for j in range(n_vertices_x - 1):
            pos = (0.25 * x[i, j, :]
                   + 0.25 * x[i + 1, j, :]
                   + 0.25 * x[i, j + 1, :]
                   + 0.25 * x[i + 1, j + 1, :])
            pos.shape = (1, 2)
            element_centers[i, j, :] = pos
    return element_centers

def plot_displacement(n_vertices_x, n_vertices_y, element_width, element_height, displacements, densities, save_path, save_postfix):
    x_undef = compute_x_undef(n_vertices_x, n_vertices_y, element_width, element_height)
    u = np.reshape(displacements, (x_undef.shape[0], x_undef.shape[1], x_undef.shape[2]))
    x = x_undef + u
    element_centers = compute_element_centers(n_vertices_x, n_vertices_y, x)
    fig, ax = plt.subplots()
    ax.plot(x[:,:,0], x[:,:,1], 'k')
    ax.plot(x[:,:,0].T, x[:,:,1].T, 'k')
    plt.scatter(element_centers[:,:,0], element_centers[:,:,1], c = densities)
    fig.savefig(os.path.join(save_path, 'displacement' + save_postfix + '.png'))
    plt.close(fig)



def estimate_compliance(problem, densities, save_path, save_postfix, problem_bc_args={}):
    assert len(densities.shape) == 2
    assert densities.shape[0] > 1 and densities.shape[1] > 1

    print('Estimating compliance ', end='', flush=True)
    domain = problem.domain.copy()
    n_vertices_x = densities.shape[1] + 1
    n_vertices_y = densities.shape[0] + 1
    n_dofs = n_vertices_y * n_vertices_x * 2
    element_densities = densities
    density_exponent = 1.0
    E_max = 1.0
    E_min = 1e-8

    assert domain[0] == 0.0
    assert domain[2] == 0.0
    element_width = domain[1] / (n_vertices_x - 1)
    element_height = domain[3] / (n_vertices_y - 1)

    n_vertices = n_vertices_x  * n_vertices_y
    n_elements = (n_vertices_x - 1 ) * (n_vertices_y - 1)

    fixed = np.full((n_vertices_y, n_vertices_x, 2), False, dtype=np.bool)
    forces = np.full((n_vertices_y, n_vertices_x, 2), 0.0, dtype=np.float64)
    problem.fem_bcs(n_vertices_x, n_vertices_y, element_width, element_height,
        fixed, forces=forces, **problem_bc_args)

    fixed = np.reshape(fixed, (n_dofs, ))
    dat = np.zeros((n_elements * 8 * 8, ), dtype=forces.dtype)
    row = np.zeros((n_elements * 8 * 8, ), dtype=np.int32)
    col = np.zeros((n_elements * 8 * 8, ), dtype=np.int32)
    element_youngs_moduli = E_min + np.power( element_densities, density_exponent) * (E_max - E_min)
    element_youngs_moduli = np.reshape(element_youngs_moduli, (n_elements, ))

    print('Computing stiffness ', end='', flush=True)
    end_idx = compute_stiffness_matrix(element_width, element_height,
        n_vertices_x, n_vertices_y, fixed, element_youngs_moduli,
        dat, row, col)

    right_hand_side = np.reshape(forces, (n_dofs, 1))

    print('Solving linear system ', end='', flush=True)
    stiffness_matrix = sp.csc_matrix((dat[0:end_idx], (row[0:end_idx], col[0:end_idx])), shape=(2 * n_vertices, 2 * n_vertices) )
    use_cvxopt = True
    if use_cvxopt:
        K = stiffness_matrix.tocoo()
        K = cvxopt.spmatrix(K.data, K.row.astype(np.int), K.col.astype(np.int))
        displacements = cvxopt.matrix(np.reshape(forces, (n_dofs, 1)), (n_dofs, 1), 'd' )
        cholmod.linsolve(K, displacements)
    else:
        displacements = spsolve(stiffness_matrix, right_hand_side)

    displacements = np.reshape(np.array(displacements), (n_dofs, 1))
    print('done', flush=True)
    plot_displacement(n_vertices_x, n_vertices_y, element_width, element_height, displacements, densities, save_path, save_postfix)
    
    comp = np.dot(displacements.T, stiffness_matrix * displacements)
    

    return comp.astype(np.float32).item()
