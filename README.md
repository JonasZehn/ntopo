# NTopo: Mesh-free Topology Optimization using Implicit Neural Representations
[Paper on arxiv](https://arxiv.org/abs/2102.10782)

This repository is the official implementation of [NTopo](https://arxiv.org/abs/2102.10782).  For the purpose of understandability, this is a reimplementation of the ideas which were used to generate the results in the paper.
NTopo is mesh-free topology optimization solver using neural representations.

Comparisons with FEM were generated using a variation of the code `topopt_cholmod` available from [DTU](https://www.topopt.mek.dtu.dk/Apps-and-software/Topology-optimization-codes-written-in-Python).

## Running the code

### Requirements

This code has been tested with tensorflow 2.3, python 3.7 and CUDA 10.1 .

The complete list of packages required is shown in [requirements.txt](requirements.txt).
After setting up a python environment and installing the appropriate tensorflow requirements such as CUDA, the packages can be installed using

```
pip install -r requirements.txt
```

To see the usage of the main program run the command 

```
python run.py --help
```


## Training

Running

```
python run.py list_problems
```
will list the available problems that are not related to solution spaces
```
python run.py train Beam2D
```
generates a config file and runs it. Results will be stored in a subfolder `./results/<experiment>`.

A modified config file can be run with
```
python run.py train_config <config_file>
```

### Evaluation
To run the evaluation, one can run
```
python run.py evaluate <folder> <density_weights>
```
In practice, this could look something like
```
python run.py evaluate results/Beam2DSpaceVolume-Adam-###_vol_#.# density_model-000100 --n_q_samples=100
````
which will create density images and a `data.json` file in a new results folder.

## Training a solution space
Running
```
python run.py list_problems_space
```
will list problems related to solution spaces.
Similarly, other commands are available with a `_space` suffix to train and evaluate models.

## License
See the [LICENSE](LICENSE) file for license rights and limitations.
