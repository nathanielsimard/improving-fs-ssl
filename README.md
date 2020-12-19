# Improving Few-Shot Learning with Auxiliary Self-Supervised Pretext Tasks

## Installing 

You need to have anaconda installed to run this repository.
Miniconda does the job perfectly.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

We provided a file containing all dependencies required to run any experiments.
We recommend using it inside a new virtual environment.

```bash
conda env create -n improving-fs-ssl
conda env update -f environement.yml
```

Some development libraries may also be useful.

```bash
conda env update -f environment-dev.yml
```

Finally you have to install the current  project.

```bash
pip install -e .
```

## Experiments

### Notebook

It is possible to run experiments with collab by using the notebook `notebook/improving-fs-ssl.ipynb`.

### Training

Experiments are launched via `./scripts/train.py`.
To reproduce some experiments, e.g `CIFAR-FS - Sup. + BYOL`, you can specified a combination of multiple configuration files that are provided under the `config` directory.

```bash
./scripts/train.py -c config/cifar_fs.yml config/supervised_byol.yml config/seed_1.yml -o /path/to/trained/experiment/
```

It is also possible to create your own configuration file to override some defaults, all defined in `mcp/config/parser.py`.

### Evaluation

Evaluation are simply ran using the script `./scripts/eval.py`.
You simply need to provide the path to a trained experiment and the evaluation configuration will be read automatically.

```bash
./scripts/eval.py -r /path/to/trained/experiment/
```

### Visualize

To plot some default visualizations and get some results, you may run the script `./scripts/viz.py` similar to evaluation.
The plots will be saved under the path provided.

```bash
./scripts/viz.py -r /path/to/trained/experiment/
```
