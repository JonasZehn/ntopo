
import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ntopo.monitors import SimulationMonitor
from ntopo.physics import compute_elasticity_energies, compute_opt_energy, compute_volume_penalty
from ntopo.filter import apply_sensitivity_filter
from ntopo.utils import write_to_file, get_sample_generator, get_single_random_q_sample_generator, get_q_sample_generator, stratified_sampling
from ntopo.oc import compute_oc_multi_batch
from ntopo.render import save_densities_to_file


def get_train_disp_step(opt, problem, disp_model, density_model, disp_variables):
    @tf.function
    def _train_disp_step(samples):
        with tf.GradientTape() as tape:
            tape.watch(disp_variables)
            internal_energy, force_loss = compute_elasticity_energies(
                problem, disp_model, density_model, samples, training=True)
            reg_loss = tf.keras.backend.sum(disp_model.losses)
            loss = internal_energy + force_loss + reg_loss

        dLdwx = tape.gradient(loss, disp_variables)
        opt.apply_gradients(zip(dLdwx, disp_variables))
        return loss, internal_energy, force_loss, reg_loss
    return _train_disp_step

def run_simulation(problem, disp_model, train_disp_step, n_sim_iterations, sim_sample_generator, saving=False, save_path='.', save_postfix=''):
    simulation_monitor = SimulationMonitor(n_sim_iterations)
    progress_bar = tqdm(simulation_monitor, total=n_sim_iterations)
    for disp_iter in progress_bar:
        start_time = time.time()

        input_samples = next(sim_sample_generator)

        loss, internal_energy, force_loss, reg_loss = train_disp_step(input_samples)
        simulation_monitor.monitor(loss)

        end_time = time.time()
        loss = loss.numpy().item()
        internal_energy = internal_energy.numpy().item()
        duration = end_time - start_time
        reg_loss = reg_loss.numpy().item()
        progress_bar.set_description(f'loss {loss:.3e} int. energy {internal_energy:.3e}, dur.: {duration:.3e}, reg loss {reg_loss:.3e}')
        progress_bar.refresh()

    if saving:
        simulation_monitor.save_plot(save_path, '', save_postfix)

def get_train_density_step(opt, problem, disp_model, density_model, density_variables, vol_penalty_strength, target_volume_ratio):
    sample_volume = problem.domain_volume
    target_volume = problem.free_volume * target_volume_ratio

    @tf.function
    def _train_densities_step(sample_positions):
        with tf.GradientTape() as tape:
            tape.watch(density_variables)
            energy, densities = compute_opt_energy(
                problem, disp_model, density_model, sample_positions)
            penalty = compute_volume_penalty(densities, sample_volume=sample_volume,
                                             vol_penalty_strength=vol_penalty_strength, target_volume=target_volume)
            reg_loss = tf.keras.backend.sum(density_model.losses)
            loss = energy + penalty + reg_loss

        dLdwx = tape.gradient(loss, density_variables)
        opt.apply_gradients(zip(dLdwx, density_variables))
        return loss, penalty, reg_loss
    return _train_densities_step


@tf.function
def compute_sensitivities(problem, disp_model, density_model, sample_positions, use_oc, vol_penalty_strength, target_volume_ratio=None):
    sample_volume = problem.domain_volume
    target_volume = problem.free_volume * target_volume_ratio
    with tf.GradientTape() as tape:
        energy, densities = compute_opt_energy(
            problem, disp_model, density_model, sample_positions)
        if use_oc:
            loss = energy
        else:
            penalty = compute_volume_penalty(densities, sample_volume=sample_volume,
                                             vol_penalty_strength=vol_penalty_strength, target_volume=target_volume)
            loss = energy + penalty

    old_densities = densities
    grads = tape.gradient(loss, densities)
    return old_densities, grads


@tf.function
def compute_target_densities_gradient_descent(old_densities, sensitivities):
    projected_sensitivities = [tf.math.maximum(0.0, tf.math.minimum(
        1.0, old_densities[i] - sensitivities[i])) - old_densities[i] for i in range(len(old_densities))]
    step_size = 0.05 / tf.math.reduce_mean([tf.math.reduce_mean(tf.math.abs(si))
                                    for si in projected_sensitivities])
    return [old_densities[i] - step_size * sensitivities[i] for i in range(len(old_densities))]


@tf.function
def optimize_densities_mse(opt, density_model, sample_positions, targets, density_variables):

    with tf.GradientTape() as tape:
        tape.watch(density_variables)

        err = density_model(sample_positions, training=True) - targets
        reg_loss = tf.keras.backend.sum(density_model.losses)
        reconstruction_loss = tf.reduce_mean(err*err, keepdims=True)
        loss = reconstruction_loss + reg_loss

    dLdwrho = tape.gradient(loss, density_variables)
    opt.apply_gradients(zip(dLdwrho, density_variables))
    return loss, reconstruction_loss, reg_loss


def save_model_configs(disp_model, density_model, save_path):
    write_to_file(disp_model.to_json(), os.path.join(
        save_path, 'disp_model_config.json'))
    write_to_file(density_model.to_json(), os.path.join(
        save_path, 'density_model_config.json'))


def save_model_weights(disp_model, density_model, save_path, save_postfix):
    disp_model.save_weights(os.path.join(
        save_path, 'disp_model' + save_postfix))
    density_model.save_weights(os.path.join(
        save_path, 'density_model' + save_postfix))


def train_non_mmse(problem, disp_model, density_model, opt_disp, opt_density,
                   opt_sample_generator, sim_sample_generator,
                   vol_penalty_strength,
                   target_volume_ratio,
                   save_path,
                   save_interval,
                   n_opt_iterations,
                   n_sim_iterations
    ):

    train_disp_step = get_train_disp_step(
        opt_disp, problem, disp_model, density_model=density_model,
        disp_variables=disp_model.trainable_variables)
    train_density_step = get_train_density_step(
        opt_density, problem, disp_model, density_model=density_model,
        density_variables=density_model.trainable_variables,
        vol_penalty_strength=vol_penalty_strength,
        target_volume_ratio=target_volume_ratio)

    save_model_configs(disp_model, density_model, save_path)

    def save_state(save_postfix):
        save_model_weights(disp_model, density_model, save_path, save_postfix)

        problem.plot_densities(density_model, save_path, '', save_postfix)

    iteration = 0
    saving = True
    save_postfix = f'-{iteration:06d}'
    run_simulation(problem, disp_model, train_disp_step, n_sim_iterations=n_sim_iterations,
                   sim_sample_generator=sim_sample_generator, saving=saving, save_path=save_path, save_postfix=save_postfix)
    if saving:
        problem.plot_displacement(disp_model, save_path, '', save_postfix)

    save_state(save_postfix)

    for iteration in range(1, n_opt_iterations + 1):
        print('Optimization iteration ', iteration)
        saving = (iteration % save_interval == 0)
        save_postfix = f'-{iteration:06d}'

        run_simulation(problem, disp_model, train_disp_step, n_sim_iterations=n_sim_iterations,
                       sim_sample_generator=sim_sample_generator, saving=saving, save_path=save_path, save_postfix=save_postfix)
        if saving:
            problem.plot_displacement(disp_model, save_path, '', save_postfix)

        sample_positions = next(opt_sample_generator)
        train_density_step(sample_positions)

        if saving:
            save_state(save_postfix)


def train_mmse(problem, disp_model, density_model, opt_disp, opt_density,
               opt_sample_generator, sim_sample_generator,
               n_opt_samples,
               vol_penalty_strength,
               target_volume_ratio,
               save_path,
               filter,
               filter_radius,
               use_oc,
               save_interval,
               n_opt_iterations,
               n_sim_iterations,
               n_opt_batches,
               oc_config):

    density_variables = density_model.trainable_variables

    train_disp_step = get_train_disp_step(
        opt_disp, problem, disp_model, density_model=density_model, disp_variables=disp_model.trainable_variables)

    save_model_configs(disp_model, density_model, save_path)

    def save_state(save_postfix, target_densities=None):
        save_model_weights(disp_model, density_model, save_path, save_postfix)

        problem.plot_densities(density_model, save_path, '', save_postfix)
        if target_densities is not None and problem.dim == 2:
            save_densities_to_file(np.reshape(target_densities[0], (n_opt_samples[1], n_opt_samples[0])), filename=os.path.join(
                save_path, 'density' + save_postfix + '-target0.png'))

    iteration = 0
    saving = True
    save_postfix = f'-{iteration:06d}'
    run_simulation(problem, disp_model, train_disp_step, n_sim_iterations=n_sim_iterations,
                   sim_sample_generator=sim_sample_generator, saving=True, save_path=save_path, save_postfix=save_postfix)
    if saving:
        problem.plot_displacement(disp_model, save_path, '', save_postfix)

    save_state(save_postfix)

    for iteration in range(1, n_opt_iterations + 1):
        print('Optimization iteration ', iteration)
        saving = (iteration % save_interval == 0)
        save_postfix = f'-{iteration:06d}'

        run_simulation(problem, disp_model, train_disp_step, n_sim_iterations=n_sim_iterations,
                       sim_sample_generator=sim_sample_generator, saving=saving, save_path=save_path, save_postfix=save_postfix)
        if saving:
            problem.plot_displacement(disp_model, save_path, '', save_postfix)

        old_densities = []
        sensitivities = []
        sample_positions = []

        for _ in range(n_opt_batches):
            input_samples = next(opt_sample_generator)

            old_di, sensitivities_i = compute_sensitivities(
                problem, disp_model, density_model, input_samples, use_oc, vol_penalty_strength=vol_penalty_strength, target_volume_ratio=target_volume_ratio)

            if filter == 'sensitivity':
                sensitivities_i = apply_sensitivity_filter(
                    input_samples, old_di, sensitivities_i, n_samples=n_opt_samples, domain=problem.domain, dim=problem.dim, radius=filter_radius)
            else:
                assert filter in ('none', ), 'not supported filter'

            old_densities.append(old_di)
            sensitivities.append(sensitivities_i)
            sample_positions.append(input_samples)

        if use_oc:
            target_densities = compute_oc_multi_batch(
                old_densities=old_densities, sensitivities=sensitivities, sample_volume=problem.domain_volume, target_volume=problem.free_volume * target_volume_ratio,
                max_move=oc_config['max_move'], damping_parameter=oc_config['damping_parameter'])
        else:
            target_densities = compute_target_densities_gradient_descent(
                old_densities=old_densities, sensitivities=sensitivities)

        progress_bar = tqdm(range(n_opt_batches))
        for i in progress_bar:
            loss, reconstruction_loss, reg_loss = optimize_densities_mse(
                opt_density, density_model, sample_positions[i], target_densities[i], density_variables)
            loss = loss.numpy().item()
            reconstruction_loss = reconstruction_loss.numpy().item()
            reg_loss = reg_loss.numpy().item()
            progress_bar.set_description(f'loss {loss} rec. loss {reconstruction_loss} reg loss {reg_loss}')
            progress_bar.refresh()

        if saving:
            save_state(save_postfix, target_densities)


def train_mmse_space(problem, disp_model, density_model, opt_disp, opt_density,
                     n_sim_samples, n_opt_samples,
                     opt_sample_generator,
                     vol_penalty_strength,
                     target_volume_ratio,
                     save_path,
                     filter,
                     filter_radius,
                     use_oc,
                     save_interval,
                     n_opt_iterations,
                     n_sim_iterations,
                     n_opt_batches,
                     n_q_samples,
                     oc_config):

    density_variables = density_model.trainable_variables

    train_disp_step = get_train_disp_step(
        opt_disp, problem, disp_model, density_model=density_model, disp_variables=disp_model.trainable_variables)

    save_model_configs(disp_model, density_model, save_path)

    def save_state(save_postfix, target_densities=None):
        disp_model.save_weights(os.path.join(
            save_path, 'disp_model' + save_postfix))
        density_model.save_weights(os.path.join(
            save_path, 'density_model' + save_postfix))

        problem.plot_densities(density_model, save_path, '', save_postfix)

    iteration = 0
    saving = True
    save_postfix = f'-{iteration:06d}'
    sim_sample_generator = get_single_random_q_sample_generator(problem.q_domain, problem.domain, n_sim_samples)
    run_simulation(problem, disp_model, train_disp_step, n_sim_iterations=2*n_sim_iterations,
                   sim_sample_generator=sim_sample_generator, saving=saving, save_path=save_path, save_postfix=save_postfix)
    if saving:
        qs = stratified_sampling(problem.q_domain, n_cells=[
                                 n_q_samples],  n_points_per_cell=1, dtype=np.float32).flatten()
        for q in qs:
            save_postfix_q = f'-{iteration:06d}-q={q:.6f}'
            print('q', q)
            problem.plot_displacement(
                disp_model, save_path, '', save_postfix_q, q=np.array([[q]]))

    save_state(save_postfix)

    for iteration in range(1, n_opt_iterations + 1):
        print('Optimization iteration ', iteration)
        saving = (iteration % save_interval == 0)

        print('saving', saving)

        target_samples_all_q = []
        target_densities_all_q = []

        qs = stratified_sampling(problem.q_domain, n_cells=[
                                 n_q_samples],  n_points_per_cell=1, dtype=np.float32).flatten()

        for q in qs:
            save_postfix_q = f'-{iteration:06d}-q={q:.6f}'

            if problem.volume_ratio_q_idx != -1:
                assert problem.volume_ratio_q_idx == 0
                target_volume_ratio = q

            old_densities = []
            sensitivities = []
            sample_positions_with_q = []

            sim_sample_generator = get_q_sample_generator(
                q, problem.domain, n_samples=n_sim_samples)
            run_simulation(problem, disp_model, train_disp_step, n_sim_iterations=n_sim_iterations,
                           sim_sample_generator=sim_sample_generator, saving=saving, save_path=save_path, save_postfix=save_postfix_q)
            if saving:
                problem.plot_displacement(
                    disp_model, save_path, '', save_postfix_q, q=np.array([[q]]))

            for _ in range(n_opt_batches):
                input_samples = next(opt_sample_generator)
                q_vec = np.ones((np.prod(n_opt_samples), 1), dtype=np.float32) * q
                input_samples_with_q = np.concatenate(
                    (input_samples, q_vec), axis=1)

                old_di, sensitivities_i = compute_sensitivities(
                    problem, disp_model, density_model, input_samples_with_q, use_oc, vol_penalty_strength=vol_penalty_strength, target_volume_ratio=target_volume_ratio)
                if filter == 'sensitivity':
                    sensitivities_i = apply_sensitivity_filter(
                        input_samples, old_di, sensitivities_i, n_samples=n_opt_samples, domain=problem.domain, dim=problem.dim, radius=filter_radius)
                else:
                    assert filter in ('none', ), 'not supported filter'

                old_densities.append(old_di)
                sensitivities.append(sensitivities_i)
                sample_positions_with_q.append(input_samples_with_q)

            if use_oc:
                target_densities = compute_oc_multi_batch(
                    old_densities, sensitivities, sample_volume=problem.domain_volume, target_volume=problem.free_volume * target_volume_ratio,
                    max_move=oc_config['max_move'], damping_parameter=oc_config['damping_parameter'])
            else:
                target_densities = compute_target_densities_gradient_descent(
                    old_densities=old_densities, sensitivities=sensitivities)

            target_samples_all_q.append(sample_positions_with_q)
            target_densities_all_q.append(target_densities)

        n_batch = len(target_samples_all_q) * len(target_samples_all_q[0])
        n_samples_total = n_batch * np.prod(n_opt_samples)
        target_samples_all_q = tf.reshape(
            target_samples_all_q, [n_samples_total, problem.dim + problem.q_dim])
        target_densities_all_q = tf.reshape(
            target_densities_all_q, [n_samples_total, 1])

        indices = np.arange(n_samples_total)
        np.random.shuffle(indices)
        n_per_batch = n_samples_total // n_batch

        progress_bar = tqdm(range(n_batch))
        for i in progress_bar:
            batch_samples = tf.gather(target_samples_all_q, tf.constant(
                indices[i*n_per_batch:(i+1)*n_per_batch]), axis=0)
            batch_densities = tf.gather(target_densities_all_q, tf.constant(
                indices[i*n_per_batch:(i+1)*n_per_batch]), axis=0)
            loss, reconstruction_loss, reg_loss = optimize_densities_mse(
                opt_density, density_model, batch_samples, batch_densities, density_variables)
            loss = loss.numpy().item()
            reconstruction_loss = reconstruction_loss.numpy().item()
            reg_loss = reg_loss.numpy().item()
            progress_bar.set_description(f'loss {loss} rec. loss {reconstruction_loss} reg loss {reg_loss}')
            progress_bar.refresh()

        if saving:
            save_postfix = f'-{iteration:06d}'
            save_state(save_postfix, target_densities)
