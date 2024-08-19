
<div align="center">

<h2 style="font-size: 2.5em;">Neural Residuals for Fault Diagnosis of Dynamic Systems</h2>

[![python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.2-blue.svg)](https://pytorch.org/)
[![contributions](https://img.shields.io/badge/Contributions-welcome-297D1E)](#contributing)
[![license](https://img.shields.io/badge/License-Apache%202.0-2F2F2F.svg)](LICENSE)
<br>
[![Docker Status](https://github.com/westny/neural-residual/actions/workflows/docker-image.yml/badge.svg)](.github/workflows/docker-image.yml)
[![Apptainer Status](https://github.com/westny/neural-residual/actions/workflows/apptainer-image.yml/badge.svg)](.github/workflows/apptainer-image.yml)
[![Conda Status](https://github.com/westny/neural-residual/actions/workflows/conda.yml/badge.svg)](.github/workflows/conda.yml)

</div>

This repository contains code for training neural networks to predict residuals for fault diagnosis of dynamic systems.
It is the result of several research projects at Linköping University and the Vehicular Systems research group.
The code is designed to be easily modifiable and extendable, allowing users to experiment with different network architectures and training strategies.
At the core of the model-building process is the adoption of [neural ordinary differential equations](https://arxiv.org/abs/1806.07366) (neural ODEs) for modeling dynamic equations within the system.
We supply a baseline model that can be used to train and evaluate the residuals on our open combustion engine dataset, but the code can be easily modified to work with other datasets and models.
<br>
The code relies heavily on [<img alt="Pytorch logo" src=https://github.com/westny/dronalize/assets/60364134/b6d458a5-0130-4f93-96df-71374c2de579 height="12">PyTorch](https://pytorch.org/docs/stable/index.html) and [<img alt="Lightning logo" src=https://github.com/westny/dronalize/assets/60364134/167a7cbb-8346-44ac-9428-5f963ada54c2 height="16">PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for its functionality.


<p align="center">
  <img width="900" src=https://github.com/user-attachments/assets/9a46d291-c57e-464f-8bcd-b06734499383>
  <br>
  <em> Visualization of ground truth and predicted signals on test data using a trained model.</em>
</p>

***

- [Installation](#installation)
  - [Apptainer](#apptainer)
  - [Docker](#docker)
  - [Conda](#conda)
  - [Pypi](#pypi)
- [Dataset](#dataset)
- [Usage](#usage)
- [Related Work](#related-work)
- [Cite](#cite)
- [Contributing](#contributing)

## Installation
There are several alternatives to installation, depending on your needs and preferences.
Our recommendation and personal preference is to use containers for reproducibility and consistency across different environments.
We have provided both an Apptainer .def file and a Dockerfile for this purpose.
Both recipes use the `mamba` package manager for creating the environments. 
In both cases, they utilize the same `environment.yml` file that could also be used to create a local conda environment if desired.
Additionally, we provide a `requirements.txt` file for those who prefer to use `pip` for package management.
All necessary files to install the required dependencies are found in the [build](build) directory.

### <img alt="Apptainer logo" src=https://github.com/westny/dronalize/assets/60364134/6a9e51ae-c6ce-4ad1-b79f-05ca7d959062 width="110">
<a id="apptainer"></a>

[Apptainer](https://apptainer.org/docs/user/main/index.html) is a lightweight containerization tool that we prefer for its simplicity and ease of use.
Once installed, you can build the container by running the following command:

```bash
apptainer build pyresidual.sif /path/to/definition_file
```

where `/path/to/definition_file` is the path to the `apptainer.def` file in the repository.
Once built, it is very easy to run the container as it only requires a few extra arguments. 
For example, to start the container and execute the `train.py` script, you can run the following command from the repository root directory:

```bash
apptainer run /path/to/pyresidual.sif python train.py
```

If you have CUDA installed and want to use GPU acceleration, you can add the `--nv` flag to the `run` command.

```bash
apptainer run --nv /path/to/pyresidual.sif python train.py
```

### <img alt="Docker logo" src=https://github.com/westny/dronalize/assets/60364134/1bf2df76-ab44-4bae-9623-03710eff0572 width="100">
<a id="docker"></a>
If you prefer to use [Docker](https://www.docker.com/get-started/), you can build the container by running the following command from the build directory:

```bash 
docker build -t pyresidual .
```

This will create a Docker image named `pyresidual` with all the necessary dependencies.
To run the container, you can use the following command:

```bash
docker run -it pyresidual
```

Note that training using docker requires mounting the data directory to the container.
Example of how this is done from the repository root directory:

```bash
docker run -v "$(pwd)":/app -w /app pyresidual python train.py
```

To use GPU acceleration, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Add the NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
and run the container with the `--gpus all` flag.

```bash
docker run --gpus all -v "$(pwd)":/app -w /app pyresidual python train.py
```

### <img alt="Conda logo" src=https://github.com/westny/dronalize/assets/60364134/52d02aa9-6231-4261-8e0f-6c092991c89c width="100">
<a id="conda"></a>
If you prefer to not use containers, you can create a [conda](https://conda.io/projects/conda/en/latest/index.html) environment using the `environment.yml` file.
To create the environment, run the following command:

```bash
conda env create -f /path/to/environment.yml
```
or if using [mamba](https://mamba.readthedocs.io/en/latest/)
    
```bash
mamba env create -f /path/to/environment.yml
```

This will create a new conda environment named `pyresidual` with all the necessary dependencies.
Once the environment is created, you can activate it by running:

```bash
conda activate pyresidual
```

The environment is now ready to use, and you can run the scripts in the repository.

To deactivate the environment, run:

```bash
conda deactivate
```

### <img alt="Pypi logo" src=https://res.cloudinary.com/practicaldev/image/fetch/s--4-K6Sjm4--/c_imagga_scale,f_auto,fl_progressive,h_420,q_auto,w_1000/https://cdn-images-1.medium.com/max/1600/1%2A_Wkc-WkNu6GJAAQ26jXZOg.png width="100">
<a id="pypi"></a>
You also have the option to install the necessary libraries using `pip` using the `requirements.txt` file.
We recommend using a virtual environment to avoid conflicts with other packages.
First, create a new virtual environment using `venv`:

```bash
python3.x -m venv pyresidual
```
where `x` is the version of Python you are using, e.g., `3.11` (used in the containers).

Activate the virtual environment:
```bash
source pyresidual/bin/activate
```

Then install the required packages using `pip`:
```bash
pip install -r /path/to/requirements.txt
```

The environment is now ready to use, and you can run the scripts in the repository.

To deactivate the virtual environment, run:

```bash
deactivate
```

Anytime you want to use the environment, you need to activate it again.

<br>

## Dataset

For the combustion engine dataset, we use the _no-fault_ data from the [LiU-ICE Industrial Fault Diagnosis Benchmark](https://vehsys.gitlab-pages.liu.se/diagnostic_competition/) available [here](https://vehsys.gitlab-pages.liu.se/diagnostic_competition/).
Upon execution, the code will download the dataset and preprocess it by default.

<p align="center">
  <img height="380" src=https://github.com/westny/neural-stability/assets/60364134/33ae2cd4-56f5-434b-a97c-9e5d8ed3ccce style="margin-right: 20px;"> <!-- Added margin-right -->
  <img height="380" src=https://github.com/westny/neural-stability/assets/60364134/8ba77fdb-235b-4cc5-a8d1-b3b69f94bc1a>
  <br>
  <em>The figure on the left depicts the schematic of the air path through the engine (test bench shown on the right).</em>
</p>

### Data Loading
In [datamodules](datamodules), you will find the necessary functionality for processing and loading the data into PyTorch training pipelines.
It includes:
- `process_data`: A function for (down)loading the data. Found in: [process.py](datamodules/process.py)
- `MTSDataset`: A `Dataset` class. Found in: [dataset.py](datamodules/dataset.py)
- `LitDataModule`: A `DataModule` class, including `Dataloader` built around `lightning.pytorch`. Found in: [dataloader.py](datamodules/dataloader.py)

<br>

## Usage
The repository is designed to be easily modifiable and extendable, allowing users to experiment with different network architectures and training strategies.
The complete model is governed by three modules/equations:
```math
  \begin{aligned}
    z_0 &\sim g_\phi(z_0 | u) \\
    \dot{z} &= f_\gamma(z, u) \\
    y &= h_\theta(z, u) \\
  \end{aligned}
```
that includes:
1. the (optional) encoder model $g_\phi$, for the variational autoencoder (VAE) approach.
2. The dynamic model $f_\gamma$, parameterized using a neural ordinary differential equation (neural ODE) by default.
3. The prediction model $h_\theta$, used to predict the residuals.

Importantly, all modules are modeled using neural networks with learnable parameters $\phi, \gamma$ and $\theta$.
The emphasis of our work is the modeling of the dynamics, which can be done using various architectures.
By default, the dynamic model is a neural ODE, but it can be easily modified to use other architectures.
As reference, we provide other alternatives, such as RNNs and Transformer-based models.

### Modeling
All neural network modules are found in the [models](models) directory.
Most of the modules are all based on an easily modifiable multi-layer perceptron (MLP) architecture, `FeedForward` found [models/layers/feedforward.py](models/layers/feedforward.py).
The architecture is designed to be modified using standalone configuration files, which are loaded into the model classes.

```python
# feedforward.py
import warnings
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 num_hidden: int = 8,
                 num_layers: int = 1,
                 activation_fn: str = "relu",
                 dropout_prob: float = 0.0,
                 norm_layer: str = "none",
                 **kwargs) -> None:
        super().__init__()

        self.net = self._create_net(num_inputs,
                                    num_outputs,
                                    num_hidden,
                                    num_layers,
                                    activation_fn,
                                    dropout_prob,
                                    norm_layer)

    @staticmethod
    def _create_net(num_inputs: int,
                    num_outputs: int,
                    num_hidden: int,
                    num_layers: int,
                    activation_fn: str = "relu",
                    dropout_prob: float = 0.0,
                    norm_layer: str = "none") -> nn.Sequential:
        def block(in_features: int, out_features: int) -> list:

            layers = [nn.Linear(in_features, out_features)]

            match norm_layer:
                case "layer":
                    layers.append(nn.LayerNorm(out_features))
                case "batch":
                    layers.append(nn.BatchNorm1d(out_features))
                case "group":
                    assert out_features % 2 == 0, "GroupNorm requires even number of features."
                    layers.append(nn.GroupNorm(out_features // 2, out_features))
                case "none":
                    pass
                case _:
                    warnings.warn(f"Unknown normalization layer: {norm_layer}. Using no normalization.")
                    pass

            if dropout_prob:
                layers.append(nn.Dropout(dropout_prob))

            match activation_fn:
                case "relu":
                    layers.append(nn.ReLU(inplace=True))
                case "lrelu":
                    layers.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))
                case "elu":
                    layers.append(nn.ELU(inplace=True))
                case "silu":
                    layers.append(nn.SiLU(inplace=True))
                case "gelu":
                    layers.append(nn.GELU(approximate="none"))
                case "tanh":
                    layers.append(nn.Tanh())
                case "sigmoid":
                    layers.append(nn.Sigmoid())
                case "none":
                    # Will return a linear model
                    pass
                case _:
                    warnings.warn(f"Unknown activation function: {activation_fn}. Using ReLU.")
                    layers.append(nn.ReLU(inplace=True))

            return layers

        net = nn.Sequential(
            # input layer
            *block(num_inputs, num_hidden),

            # hidden layers
            *[module for _ in range(num_layers - 1) for
              module in block(num_hidden, num_hidden)],

            # output layer
            nn.Linear(num_hidden, num_outputs),
        )

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

```

### Model Training
The repository includes a training script, [train.py](train.py), that can be used to train your models on the preprocessed data.
The script is designed to be run from the repository root directory and includes several arguments that can be used to configure the training process.

#### Configuration
We employ two separate configuration files in YAML format for the model and training hyperparameters, which can be modified to suit your needs.
In [configs/networks](configs/networks), you will find the model configuration files, detailing the required modules and hyperparameters for training.
Next, in [configs/residuals](configs/residuals), you will find the residual configuration files, detailing the dynamic equations and prediction targets for the residuals.
Both of these files are loaded into the training script and used to build the model and training loop.
See [configs/README.md](configs/README.md) for more information on the configuration files.

Additional runtime arguments, such as the number of workers, GPU acceleration, debug mode, and model checkpointing, can be specified when running the script (see [arguments.py](arguments.py) for more information).

#### Running the Training Script
The training script is designed to be used with PyTorch Lightning; besides using the custom data modules previously mentioned, it also requires a `LightningModule` that defines the model and training loop.
In [models/litmodule.py](models/litmodule.py), you will find a base class that can be modified to build your own `LightningModule`. 
In its current form, it can be used to train and evaluate models defined by the supplied configuration files.

An example of how to train the model is shown below:
```bash
  [apptainer run --nv path/to/pyresidual.sif] python train.py --add-name test --config neuralode.yml --residual greybox.yml
```

We recommend users modify the default arguments in [arguments.py](arguments.py) to suit their needs.

> Note that the default logger is set to `wandb` ([weights & biases](https://wandb.ai/)) for logging performance metrics during training.
> It is our preferred tool for tracking experiments, but it can be easily replaced with other logging tools by modifying the `Trainer` in the training script.
> 
> See the official [Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for more information on customizing training behavior and how to use the library in general.


<br>

## Related work
We have been working with the engine data and neural residual generation in several research projects, resulting in multiple published papers.
If you're interested in learning more about our findings, please refer to the following publications:

#### [Automated Design of Grey-Box Recurrent Neural Networks For Fault Diagnosis using Structural Models and Causal Information](https://proceedings.mlr.press/v168/jung22a.html)
- **Authors:** Daniel Jung
- **Published In:**  Proceedings of The 4th Annual Learning for Dynamics and Control Conference, PMLR, 2022.

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
    Behavioral modeling of nonlinear dynamic systems for control design and system monitoring of technical systems is a non-trivial task.
    One example is fault diagnosis where the objective is to detect abnormal system behavior due to faults at an early stage and isolate the faulty component. Developing sufficiently accurate models for fault diagnosis applications can be a time-consuming process which has motivated the use of data-driven models and machine learning.
    However, data-driven fault diagnosis is complicated by the facts that faults are rare events, and that it is not always possible to collect data that is representative of all operating conditions and faulty behavior.
    One solution to incomplete training data is to take into consideration physical insights when designing the data-driven models.
    One such approach is grey-box recurrent neural networks where physical insights about the monitored system are incorporated into the neural network structure.
    In this work, an automated design methodology is developed for grey-box recurrent neural networks using a structural representation of the system.
    Data from an internal combustion engine test bench is used to illustrate the potentials of the proposed network design method to construct residual generators for fault detection and isolation.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @inproceedings{jung2022automated,
      title={Automated design of grey-box recurrent neural networks for fault diagnosis using structural models and causal information},
      author={Jung, Daniel},
      booktitle={Learning for Dynamics and Control Conference},
      pages={8--20},
      year={2022},
      organization={PMLR}
    }

</details>

#### [Analysis of Numerical Integration in RNN-Based Residuals for Fault Diagnosis of Dynamic Systems](https://arxiv.org/abs/2305.04670)
- **Authors:** Arman Mohammadi, Theodor Westny, Daniel Jung, and Mattias Krysander
- **Published In:** IFAC-PapersOnLine, Vol. 56, No. 2, 2023.

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
    Data-driven modeling and machine learning are widely used to model the behavior of dynamic systems.
    One application is the residual evaluation of technical systems where model predictions are compared with measurement data to create residuals for fault diagnosis applications.
    While recurrent neural network models have been shown capable of modeling complex non-linear dynamic systems, they are limited to fixed steps discrete-time simulation.
    Modeling using neural ordinary differential equations, however, make it possible to evaluate the state variables at specific times, compute gradients when training the model and use standard numerical solvers to explicitly model the underlying dynamic of the time-series data.
    Here, the effect of solver selection on the performance of neural ordinary differential equation residuals during training and evaluation is investigated.
    The paper includes a case study of a heavy-duty truck's after-treatment system to highlight the potential of these techniques for improving fault diagnosis performance.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @article{mohammadi2023analysis,
      title={Analysis of numerical integration in RNN-based residuals for fault diagnosis of dynamic systems},
      author={Mohammadi, Arman and Westny, Theodor and Jung, Daniel and Krysander, Mattias},
      journal={IFAC-PapersOnLine},
      volume={56},
      number={2},
      pages={2909--2914},
      year={2023},
      publisher={Elsevier}
    }
</details>

#### [Stability-Informed Initialization of Neural Ordinary Differential Equations](https://arxiv.org/abs/2311.15890)
- **Authors:** Theodor Westny, Arman Mohammadi, Daniel Jung, and Erik Frisk
- **Published In:** 2024 International Conference on Machine Learning (ICML)

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
    This paper addresses the training of Neural Ordinary Differential Equations (neural ODEs), and in particular explores the interplay between numerical integration techniques, stability regions, step size, and initialization techniques.
    It is shown how the choice of integration technique implicitly regularizes the learned model, and how the solver's corresponding stability region affects training and prediction performance.
    From this analysis, a stability-informed parameter initialization technique is introduced.
    The effectiveness of the initialization method is displayed across several learning benchmarks and industrial applications.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @article{westny2023stability,
      title={Stability-Informed Initialization of Neural Ordinary Differential Equations},
      author={Westny, Theodor and Mohammadi, Arman and Jung, Daniel and Frisk, Erik},
      journal={arXiv preprint arXiv:2311.15890},
      year={2023}
    }
</details>

<details>
  <summary>Code</summary>
    
[Github link](https://github.com/westny/neural-stability)
</details>

## Cite
If you find the contents of this repository helpful, please consider citing the papers mentioned in the [related work](#related-work) section.


## Contributing
We welcome contributions to the toolbox, and we encourage you to submit pull requests with new features, bug fixes, or improvements.
Any form of collaboration is appreciated, and we are open to suggestions for new features or changes to the existing codebase.

Feel free to [email us](mailto:theodor.westny@liu.se) if you have any questions or notice any issues with the code.
If you have any suggestions for improvements or new features, we would be happy to hear from you.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
