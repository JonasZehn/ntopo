"""NTopo.

Usage:
  run.py list_problems
  run.py list_problems_space
  run.py train_config <config_file>
  run.py train <problem> [--volume_ratio=0.5] [--lr_sim=3e-4] [--lr_opt=3e-4]
    [--save_interval=5] [--use_mmse=1] [--filter=sensitivity] [--filter_radius=2.0]
    [--vol_penalty_strength=10]
    [--use_oc=1] [--n_opt_iterations=1000] [--n_sim_iterations=1000] [--n_opt_batches=50]
  run.py evaluate <folder> <weights_name> [--n_samples=<n>]
    [--n_samples_compliance=<n>]
  run.py train_space_config <config_file>
  run.py train_space <problem> [--volume_ratio=0.5] [--lr_sim=3e-4] [--lr_opt=3e-4]
    [--save_interval=5] [--filter=sensitivity]  [--filter_radius=2.0] [--use_oc=1]
    [--n_opt_iterations=1000] [--n_sim_iterations=1000] [--n_opt_batches=50]
    [--n_q_samples=5]
  run.py evaluate_space <folder> <weights_name> [--n_samples=<n>]
    [--n_samples_compliance=<n>] [--n_q_samples=<kn>]
  run.py --version
  run.py --help

Options:
  -h --help                      Show this screen.
  --version                      Show version.
  --vol_penalty_strength=<sps>   Penalty weight when not using OC [default: 10]
  --volume_ratio=<vr>            Target volume ratio [default: 0.5]
  --lr_sim=<lrs>                 Learning rate for the displacement field [default: 3e-4]
  --lr_opt=<lro>                 Learning rate for the density field [default: 3e-4]
  --save_interval=<sv>           Save interval during optimization [default: 5]
  --use_mmse=<mmse>              Whether to use mmse for training [default: 1]
  --filter=<filter>              Filter to use, either none, sensitivity [default: sensitivity]
  --filter_radius=<sfr>          Filter radius [default: 2.0]
  --use_oc=<oc>                  Use optimality criterion, either 0 or 1 [default: 1]
  --n_opt_iterations=<no>        Number of optimization iterations for outer loop [default: 1000]
  --n_sim_iterations=<ns>        Number of inner simulation steps [default: 1000]
  --n_opt_batches=<nb>           Number of batches for updating the density network [default: 50]
  --n_q_samples=<kn>             Number of samples of q [default: 5]
  --n_samples=<n>                Number of samples for creating image [default: 30000]
  --n_samples_compliance=<n>     Number of samples when evaluating compliance [default: 30000]
"""

import os
from importlib import reload

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
import tensorflow as tf

from ntopo.problems import problems, space_problems, get_problem_by_name, get_space_problem_by_name
from ntopo.models import DensityModel, DispModel
from ntopo.train import train_mmse, train_non_mmse, train_mmse_space
from ntopo.utils import (
    set_random_seed, get_sample_generator, get_default_sample_counts,
    write_to_json_file, read_from_json_file, write_to_file, get_grid_centers
)
from ntopo.render import predict_densities_image_2d, save_densities_to_file
from ntopo.fem_sim import estimate_compliance

DEFAULT_N_SIM_SAMPLES_2D = 150*50
DEFAULT_N_OPT_SAMPLES_2D = 150*50
DEFAULT_N_SIM_SAMPLES_3D = 80*40*20
DEFAULT_N_OPT_SAMPLES_3D = 80*40*20

def get_new_idx():
    """
    returns new index for storing result based on the counter.txt file
    """
    count = 0
    counter_file = 'counter.txt'
    if os.path.exists(counter_file):
        with open(counter_file, 'r', encoding='utf8') as file_object:
            count = int(file_object.read().splitlines()[-1])
    with open(counter_file, 'w+', encoding='utf8') as file_object:
        file_object.write(str(count+1))
    return count


def create_new_save_folder(name_format):
    """
    creates new folder to store result and returns the path of the folder
    """

    results_folder = 'results'

    new_idx = get_new_idx()
    folder_name = name_format.format(idx=new_idx)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    new_path = os.path.join(current_dir, results_folder, folder_name)
    if not os.path.exists(os.path.join(current_dir, results_folder)):
        os.mkdir(os.path.join(current_dir, results_folder))
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path

def instantiate_optimizer(config):
    return tf.keras.optimizers.get(config)

def train_config_single(config):
    set_random_seed(config['seed'])

    problem_name = config['problem_name']

    problem = get_problem_by_name(problem_name)
    problem.init()

    opt_disp = instantiate_optimizer(config['opt_disp'])
    opt_density = instantiate_optimizer(config['opt_density'])

    optimizer_name = type(opt_density).__name__

    save_path = create_new_save_folder(
        name_format=problem_name + '-' + optimizer_name + '-{idx}_vol_' + str(config['volume_ratio']))

    write_to_json_file(config, os.path.join(save_path, "config.txt"))

    disp_model = DispModel(input_domain=problem.domain, dim=problem.dim,
                           bc=problem.bc, features=config['disp_model_features_config'], model=config['disp_model_config'])
    density_model = DensityModel(
        input_domain=problem.domain,
        dim=problem.dim,
        volume_ratio=config['volume_ratio'],
        constraint=problem.density_constraint,
        features=config['density_model_features_config'],
        model=config['density_model_config']
    )

    print('n_sim_samples ', config['n_sim_samples'])
    print('n_opt_samples ', config['n_opt_samples'])
    sim_sample_generator = get_sample_generator(problem.domain, config['n_sim_samples'])
    opt_sample_generator = get_sample_generator(problem.domain, config['n_opt_samples'])

    if config['use_mmse'] == 1:
        train_mmse(
            problem=problem, disp_model=disp_model,
            density_model=density_model, opt_disp=opt_disp, opt_density=opt_density,
            opt_sample_generator=opt_sample_generator,
            sim_sample_generator=sim_sample_generator,
            n_opt_samples=config['n_opt_samples'],
            vol_penalty_strength=config['vol_penalty_strength'],
            target_volume_ratio=config['volume_ratio'],
            save_path=save_path,
            filter=config['filter'],
            filter_radius=config['filter_radius'],
            use_oc=config['use_oc'],
            save_interval=config['save_interval'],
            n_opt_iterations=config['n_opt_iterations'],
            n_sim_iterations=config['n_sim_iterations'],
            n_opt_batches=config['n_opt_batches'],
            oc_config=config['oc']
        )
    else:
        train_non_mmse(problem, disp_model, density_model, opt_disp, opt_density,
                       opt_sample_generator, sim_sample_generator,
                       vol_penalty_strength=config['vol_penalty_strength'],
                       target_volume_ratio=config['volume_ratio'],
                       save_path=save_path,
                       save_interval=config['save_interval'],
                       n_opt_iterations=config['n_opt_iterations'],
                       n_sim_iterations=config['n_sim_iterations'],
        )

def transfer_arguments_to_config(arguments, config):
    config['volume_ratio'] = float(arguments['--volume_ratio'])
    config['filter'] = arguments['--filter']
    config['filter_radius'] = float(arguments['--filter_radius'])
    config['save_interval'] = int(arguments['--save_interval'])
    config['vol_penalty_strength'] = float(arguments['--vol_penalty_strength'])
    config['use_mmse'] = int(arguments['--use_mmse'])
    config['n_opt_iterations'] = int(arguments['--n_opt_iterations'])
    config['n_sim_iterations'] = int(arguments['--n_sim_iterations'])
    config['n_opt_batches'] = int(arguments['--n_opt_batches'])
    config['n_q_samples'] = int(arguments['--n_q_samples'])
    config['use_oc'] = arguments['--use_oc'] != '0'
    config['opt_disp'] = {
        'class_name': 'Adam',
        'config': {
            'learning_rate': float(arguments['--lr_sim']),
            'beta_2': 0.99,
        }
    }
    config['opt_density'] = {
        'class_name': 'Adam',
        'config': {
            'learning_rate':  float(arguments['--lr_opt']),
            'beta_1': 0.8,
            'beta_2': 0.9,
        }
    }

def train_single(arguments):
    """
    generates default config from arguments and runs it
    """

    problem_name = arguments['<problem>']

    problem = get_problem_by_name(problem_name)
    problem.init()

    config = {
        'problem_name': problem_name,
        'seed': 42,
        'oc': {
            'max_move': 0.2,
            'damping_parameter': 0.5
        }
    }

    omega0 = 60.0
    if problem.dim == 2:
        n_hidden = 60
        n_sim_samples = get_default_sample_counts(problem.domain, DEFAULT_N_SIM_SAMPLES_2D)
        n_opt_samples = get_default_sample_counts(problem.domain, DEFAULT_N_OPT_SAMPLES_2D)
    else:
        n_hidden = 180
        n_sim_samples = get_default_sample_counts(problem.domain, DEFAULT_N_SIM_SAMPLES_3D)
        n_opt_samples = get_default_sample_counts(problem.domain, DEFAULT_N_OPT_SAMPLES_3D)

    features_config = {'class_name': 'ConcatSineFeatures',
                       'config': {'n_input': problem.dim}}
    config['disp_model_features_config'] = features_config
    config['disp_model_config'] = {'class_name': 'DenseSIRENModel', 'config':
        {'n_input': 2*problem.dim, 'n_output': problem.dim, 'n_hidden': n_hidden, 'last_layer_init_scale': 1e-3, 'omega0': omega0}}
    config['density_model_features_config'] = features_config
    config['density_model_config'] = {'class_name': 'DenseSIRENModel', 'config': {
        'n_input': 2*problem.dim, 'n_output': 1, 'n_hidden': n_hidden, 'last_layer_init_scale': 1e-3, 'omega0': omega0}}

    transfer_arguments_to_config(arguments, config)

    config['n_sim_samples'] = n_sim_samples.copy()
    config['n_opt_samples'] = n_opt_samples.copy()

    train_config_single(config)


def evaluate_single(arguments):

    set_random_seed(42)

    folder = arguments['<folder>']
    weights_name = arguments['<weights_name>']

    config = read_from_json_file(os.path.join(folder, "config.txt"))

    problem_name = config['problem_name']
    problem = get_problem_by_name(problem_name)
    assert problem.dim != 3, "not implemented"
    density_model = DensityModel.load_from_config_file(
        os.path.join(folder, "density_model_config.json"))
    density_model.load_weights(os.path.join(
        folder, weights_name))

    save_path = create_new_save_folder(
        name_format='evaluation-{idx}-' + problem_name)

    n_samples = int(arguments['--n_samples'])
    n_samples_xy = get_default_sample_counts(problem.domain, n_samples)
    print('n_samples ', n_samples_xy)

    save_prefix = ''
    save_postfix = ''
    filename = os.path.join(save_path, save_prefix +
                            'density' + save_postfix + '.png')

    densities_im, _ = predict_densities_image_2d(density_model,
        domain=problem.domain, mirror=problem.mirror, n_samples=n_samples_xy)
    save_densities_to_file(densities_im, filename)

    # need to evaluate seperatly as estimate_compliance and fem bcs expects not mirrored densities
    n_samples_compliance = int(arguments['--n_samples_compliance'])
    n_samples_compliance_xy = get_default_sample_counts(problem.domain, n_samples_compliance)
    print('n_samples_compliance_xy ', n_samples_compliance_xy)
    densities_comp, _ = predict_densities_image_2d(density_model,
        domain=problem.domain, mirror=[False, False], n_samples=n_samples_compliance_xy)
    filename = os.path.join(save_path, save_prefix +
                            'density_comp' + save_postfix + '.png')
    save_densities_to_file(densities_comp, filename)
    compliance = estimate_compliance(problem, densities_comp, save_path, save_postfix)

    volume_ratio_estimate = np.mean(densities_im)
    volume_ratio = config['volume_ratio']
    volume_perc_error = 100.0 * abs(volume_ratio_estimate - volume_ratio)/volume_ratio
    print('Compliance: ', compliance)
    print('Volume ratio estimate: ', volume_ratio_estimate)
    print('Volume perc. error: ', volume_perc_error, '%')

    data = {
        'n_samples': n_samples_xy,
        'volume_ratio_estimate': volume_ratio_estimate.item(),
        'volume_perc_error': volume_perc_error.item(),
        'n_samples_compliance': n_samples_compliance_xy,
        'compliance': compliance
    }
    filename = os.path.join(save_path, 'data.json')
    write_to_json_file(data, filename)

def train_config_space(config):

    set_random_seed(config['seed'])

    problem_name = config['problem_name']

    problem = get_space_problem_by_name(problem_name)
    problem.init()

    opt_disp = instantiate_optimizer(config['opt_disp'])
    opt_density = instantiate_optimizer(config['opt_density'])
    optimizer_name = type(opt_density).__name__

    save_path = create_new_save_folder(
        name_format=problem_name + '-' + optimizer_name + '-{idx}_vol_' + str(config['volume_ratio'])
    )

    write_to_json_file(config, os.path.join(save_path, "config.txt"))

    disp_model_config = {
        'input_domain': config['input_domain'], 'dim': problem.dim,
        'bc': problem.bc, 'features': config['disp_model_features_config'], 'model': config['disp_sub_model_config']
    }
    disp_model = DispModel(**disp_model_config)
    density_model = DensityModel(
        input_domain=config['input_domain'],
        dim=problem.dim, volume_ratio=config['volume_ratio'],
        constraint=problem.density_constraint,
        features=config['density_model_features_config'],
        model=config['density_model_config'],
        volume_ratio_q_idx=problem.volume_ratio_q_idx
    )

    opt_sample_generator = get_sample_generator(problem.domain, config['n_opt_samples'])

    train_mmse_space(problem, disp_model, density_model, opt_disp, opt_density,
        config['n_sim_samples'], config['n_opt_samples'],
        opt_sample_generator,
        vol_penalty_strength=config['vol_penalty_strength'],
        target_volume_ratio=config['volume_ratio'],
        save_path=save_path,
        filter=config['filter'],
        filter_radius=config['filter_radius'],
        use_oc=config['use_oc'],
        save_interval=config['save_interval'],
        n_opt_iterations=config['n_opt_iterations'],
        n_sim_iterations=config['n_sim_iterations'],
        n_opt_batches=config['n_opt_batches'],
        n_q_samples=config['n_q_samples'],
        oc_config=config['oc']
    )

def train_space(arguments):
    """
    generates default configuration and then also runs it
    """

    problem_name = arguments['<problem>']

    problem = get_space_problem_by_name(problem_name)
    problem.init()

    n_input = problem.dim + problem.q_dim

    if problem.dim == 2:
        n_sim_samples = get_default_sample_counts(problem.domain, DEFAULT_N_SIM_SAMPLES_2D)
        n_opt_samples = get_default_sample_counts(problem.domain, DEFAULT_N_OPT_SAMPLES_2D)
    else:
        n_sim_samples = get_default_sample_counts(problem.domain, DEFAULT_N_SIM_SAMPLES_3D)
        n_opt_samples = get_default_sample_counts(problem.domain, DEFAULT_N_OPT_SAMPLES_3D)

    n_hidden_disp = 256
    n_hidden_density = 256
    omega0 = 60.0

    config = {
        'problem_name': problem_name,
        'seed': 42,
        'oc': {
            'max_move': 0.2,
            'damping_parameter': 0.5
        }
    }

    transfer_arguments_to_config(arguments, config)

    config['input_domain'] = np.concatenate((problem.domain, problem.q_domain), axis=0).tolist()

    features_config = {
        'class_name': 'ConcatSineFeatures',
        'config': {'n_input': n_input}
    }
    config['disp_model_features_config'] = features_config
    config['disp_sub_model_config'] = {
        'class_name': 'DenseSIRENModel', 'config': {'n_input': 2*n_input,
            'n_output': problem.dim, 'n_hidden': n_hidden_disp, 'last_layer_init_scale': 1e-3, 'omega0': omega0}
    }
    config['density_model_features_config'] = features_config
    config['density_model_config'] = {
        'class_name': 'DenseSIRENModel', 'config': {'n_input': 2*n_input,
            'n_output': 1, 'n_hidden': n_hidden_density, 'last_layer_init_scale': 1e-3, 'omega0': omega0}
    }

    assert config['disp_model_features_config']['config']['n_input'] == (len(config['input_domain'])//2)
    assert config['density_model_features_config']['config']['n_input'] == (len(config['input_domain'])//2)

    config['n_sim_samples'] = n_sim_samples.copy()
    config['n_opt_samples'] = n_opt_samples.copy()

    train_config_space(config)


def evaluate_space(arguments):

    set_random_seed(42)

    folder = arguments['<folder>']
    weights_name = arguments['<weights_name>']

    config = read_from_json_file(os.path.join(folder, "config.txt"))
    problem_name = config['problem_name']

    save_path = create_new_save_folder(
        name_format='evaluation-{idx}-' + problem_name)
    write_to_json_file(arguments, os.path.join(save_path, "arguments.txt"))

    problem = get_space_problem_by_name(problem_name)
    density_model = DensityModel.load_from_config_file(
        os.path.join(folder, "density_model_config.json"))
    density_model.load_weights(os.path.join(
        folder, weights_name))

    n_samples = int(arguments['--n_samples'])
    n_q_samples = int(arguments['--n_q_samples'])
    n_samples_compliance = int(arguments['--n_samples_compliance'])
    [n_samples_compliance_x, n_samples_compliance_y] = get_default_sample_counts(problem.domain, n_samples_compliance)
    print('n_samples_compliance', [n_samples_compliance_x, n_samples_compliance_y])

    # even width and height is required for some video  encoders
    n_samples_xy = get_default_sample_counts(problem.domain, n_samples, even = True)
    qs = problem.get_plot_samples(n_q_samples)
    filenames = []
    volume_ratio_estimates = []
    compliances = []
    estimate_compliances = True
    save_prefix=''
    for i in range(len(qs)):
        q = qs[i]
        assert np.size(q) == 1
        q0 = q.item()
        save_postfix = f'-q={q0:.6f}'

        print('Processing ', i, '/', len(qs) - 1)
        filename = os.path.join(save_path, save_prefix +
                                'density' + save_postfix + '.png')
        densities_im, _ = predict_densities_image_2d(density_model.get_model_partial_q(q),
            domain=problem.domain, mirror=problem.mirror, n_samples=n_samples_xy)

        save_densities_to_file(densities_im, filename)
        volume_ratio_estimate = np.mean(densities_im)
        volume_ratio_estimates.append(volume_ratio_estimate.item())
        filenames.append(filename)

        if estimate_compliances:
            positions_comp = get_grid_centers(problem.domain, [n_samples_compliance_x, n_samples_compliance_y])
            densities_comp = density_model.get_model_partial_q(q).predict(positions_comp, batch_size=1024)
            densities_comp = np.reshape(densities_comp, (n_samples_compliance_y, n_samples_compliance_x))
            compliance = estimate_compliance(problem, densities_comp, save_path, save_postfix, problem_bc_args={ 'q': q })
            compliances.append(compliance)

    qs = np.concatenate(qs, axis=0)
    data = { 'qs': qs.tolist(), 'volume_ratio_estimates': volume_ratio_estimates }
    if estimate_compliances:
        data['compliances'] = compliances
    filename = os.path.join(save_path, 'data.json')
    write_to_json_file(data, filename)

    assert qs.shape[1] == 1 # code assumes 1dim q for now
    qs = qs.squeeze()
    if estimate_compliances:
        fig, ax = plt.subplots()
        ax.plot(qs, compliances)
        ax.set_xlabel('Q')
        fig.savefig(os.path.join(save_path, 'compliances.png'))
        plt.close(fig)

    volume_ratio_estimates = np.array(volume_ratio_estimates, dtype=np.float32)

    fig, ax = plt.subplots()
    ax.plot(qs, volume_ratio_estimates)
    ax.set_xlabel('Q')
    ax.set_ylabel('Volume ratio')
    fig.savefig(os.path.join(save_path, 'volume_ratio_estimates.png'))
    plt.close(fig)

    if problem.volume_ratio_q_idx != -1:
        volume_ratio_rel_error = abs(volume_ratio_estimates - qs) / qs
    else:
        volume_ratio_rel_error = abs(volume_ratio_estimates - config['volume_ratio']) / config['volume_ratio']
    fig, ax = plt.subplots()
    ax.plot(qs, volume_ratio_rel_error * 100.0)
    ax.set_xlabel('Q')
    ax.set_ylabel('Volume percentage error')
    fig.savefig(os.path.join(save_path, 'volume_ratio_errors.png'))
    plt.close(fig)

    lines = [
        "file '" + os.path.split(f)[1] + "'" + '\n' + 'duration 0.03' for f in filenames]
    filenames_str = '\n'.join(lines)
    write_to_file(filenames_str, os.path.join(save_path, 'filenames.txt'))
    write_to_file('ffmpeg -safe 0 -f concat -i filenames.txt  -pix_fmt yuv420p output.mp4',
                  os.path.join(save_path, 'make_video.bat'))
    write_to_file('#!/bin/bash\nffmpeg -safe 0 -f concat -i filenames.txt -pix_fmt yuv420p -f mp4 output.mp4',
                  os.path.join(save_path, 'make_video.sh'))


def main():
    arguments = docopt(__doc__, version='NTopo 1.0')

    # in the TkAgg backend there is a bug it seems which will run into "Fail to allocate bitmap" error
    # so we use a different backend
    # http://matplotlib.1069221.n5.nabble.com/Fail-to-allocate-bitmap-Unable-to-free-colormap-palette-is-still-selected-td13203.html
    reload(matplotlib)
    matplotlib.use('Agg')

    if arguments['list_problems']:
        print('Problems:')
        for p in problems:
            print('   ', p.__name__)

    elif arguments['list_problems_space']:
        print('Space problems:')
        for p in space_problems:
            print('   ', p.__name__)
    elif arguments['train_config']:
        config = read_from_json_file(arguments['<config_file>'])
        train_config_single(config)
    elif arguments['train']:
        train_single(arguments)
    elif arguments['evaluate']:
        evaluate_single(arguments)
    elif arguments['train_space_config']:
        config = read_from_json_file(arguments['<config_file>'])
        train_config_space(config)
    elif arguments['train_space']:
        train_space(arguments)
    elif arguments['evaluate_space']:
        evaluate_space(arguments)


if __name__ == '__main__':
    main()
