# Configuration files
The configuration files are used to define the parameters of the experiments.
Stored in the `configs` directory, the configuration files are written in YAML format and are organized in two different subdirectories: `networks` and `residuals`.
The `networks` directory contains the configuration files that are used to specify the network hyperparameters and general experiment settings, such as the number of epochs, the batch size, the learning rate, etc.
The `residuals` directory contains the configuration files that are used to specify the residual hyperparameters, i.e, the number of residuals, the underlying dynamics and dataset signals to be used.

## Network Configuration
Recall that the complete model is governed by three modules/equations:
```math
  \begin{aligned}
    z_0 &\sim g_\phi(z_0 | u) \\
    \dot{z} &= f_\gamma(z, u) \\
    y &= h_\theta(z, u) \\
  \end{aligned}
```
In the configuration files, the `encoder_model` refers to $g_\phi$, the `dynamic_model` refers to $f_\gamma$, and the `predictor_model` refers to $h_\theta$.

Let us look at the basic network configuration file provided, `neuralode.yml`:


```yaml
# General Description
description: Combustion Engine Dynamics
task: MTES1
dataset: engine

# Model
model: {
  class: ResidualPredictor,
  module: models.model,
  variational: false,

  # Dynamic Model
  dynamic_model: {

    # ODE Dynamics
    step_size: 0.05,
    solver: midpoint,
    stability_init: false,
    complex_poles: false,

    # Network configuration
    net: {
      num_hidden: 128,
      num_layers: 2,
      activation_fn: silu,
      dropout_prob: 0.0,
      norm_layer: none,
    }
  },

  # Predictor Model
  predictor_model: {

    # Network configuration
    net: {
      ...
    }
  },

  # Encoder Model
  encoder_model: {

    # Network configuration
    net: {
      ...
    }

  }
}

# Lightning module
litmodule: {
  class: LitModel,
  module: models.litmodule
}

# Datamodule
datamodule: {
  class: LitDataModule,
  module: datamodules.dataloader,
  batch_size: 512,
  segment_len: 800,
  sequence_len: 400,
  test_sequence_len: -1,
  sample_time: 0.05,
  data_seed: 42,
  root: data/engine,
  train_file: wltp_NF.csv,
  test_file: wltp_NF_2.csv
}

# Trainer
training: {
  data: engine,
  epochs: 100,
  lr: 0.0001,
  beta: 1.0,
  decay: 0.999,
  sample_time: 0.05,
  clip: 1.0,
  criterion: mse
}

```
The bulk of the configuration file should be self-explanatory.
For more details on the network hyperparameters, please refer to the [models/layers/feedforward.py](../models/layers/feedforward.py) file.

### Dynamic Model
The `dynamic_model` dictionary is especially important, as it is used to define the neural ODE.
We note that the `step_size` parameter is only relevant when using stability-informed initialization (see below).
The length of integration is determined by runtime arguments within the `LitModel` class.
The `solver` parameter is used to specify the numerical solver to be used.
We rely on the `torchdiffeq` library for the ODE solvers, which include `euler`, `midpoint`, `rk4`, `dopri`, `adams`, and more.
For our own purposes, we have also added `rk3` and `heun` solvers.
Please refer to the [torchdiffeq](github.com/rtqichen/torchdiffeq) repository for more details on the solvers.

#### Variational Autoencoder
If `vae` is set to `true` (default: `false`), the network will be trained using a variational approach and the initial states of the dynamics are considered as random variables.
In this case, the model will employ the `encoder` model $g_\phi$, to infer the initial states of the dynamics using the first input instance and the model is trained using the evidence lower bound (ELBO) loss (otherwise, the initial states are set to zero).
This is the approach used in the ICML 2024 paper, [Stability-Informed Initialization of Neural Ordinary Differential Equations](https://arxiv.org/abs/2311.15890).

#### Stability-Informed Initialization
The `stability_init` parameter (default: `false`) is used to specify whether the parameters of the dynamic model $f_\gamma$ should be initialized using the stability-informed initialization approach.
Note that this is currently only implemented for a subset of the available solvers (see the code for more details).
Extensive details on the initialization method can be found in the ICML 2024 paper.

### Alternative Dynamics
We supply alternative approaches to modeling the dynamics, using either various RNNs or a causal Transformer architecture.
For your reference, we have prepared additional network configuration files, `rnn.yml` and `transformer.yml`, that can be used to specify the hyperparameters of these models.
Please refer to the [models/layers/masked_rnn.py](../models/layers/masked_rnn.py) and [models/layers/masked_tf.py](../models/layers/masked_tf.py) files for more details on their implementation.

### Data Module
There are two parameters in the `datamodule` dictionary that are important to clarify, `segment_len` and `sequence_len`.
Note that the original data consist of a single multivariate time series.
To create the training and validation datasets, the data is segmented into smaller segments of length `segment_len` that are allocated to the training and validation datasets using an 80/20 split.

To enable parallel computing, the segments are further divided into smaller sequences of length `sequence_len` that are used to create the mini-batches during training and validation.
As `torchdiffeq` assumes a specific ordering of the input data, each mini-batch has shape
```python
(sequence_len, batch_size, num_signals)
```
where `num_signals` is the number of signals in the dataset.

## Residual Configuration
The basic residual configuration file provided, `greybox.yml` should also be mostly self-explanatory.
It is important to note that the naming convention of the configuration parameters are important, as they are used to identify which signals are used in the residual blocks.
Consider the example configuration file below:

```yaml
# General Description
description: MTES1

signals: {
  intercooler_pressure: "y_p_ic",
  intercooler_temperature: "y_T_ic",
  intake_manifold_pressure: "y_p_im",
  air_mass_flow: "y_W_af",
  engine_speed: "y_omega_e",
  throttle_position: "y_alpha_th",
  wastegate_position: "y_u_wg",
  injected_fuel_mass: "y_wfc",
  ambient_temperature: "y_T_amb",
  ambient_pressure: "y_p_amb"
}

# Signals to be zeroed
zeroed_signals: null

# Dynamic Equations
dynamic: {
  m_t: {
    states: [ m_t ],
    inputs: [ y_W_af, y_alpha_th, y_omega_e ],
  },

  T_t: {
    states: [ T_t, m_t ],
    inputs: [ y_T_amb, y_p_amb ],
  }
}

# Predictors
predictors: {
  y_p_ic: {
    states: [ m_t ],
    inputs: [ ],
  },

  y_T_ic: {
    states: [ T_t, m_t ],
    inputs: [ y_p_amb, y_u_wg ],
  }
}


```

Note that the values of the keys under the `signals` dictionary are used to identify the signals in the dataset.
The exact names appear as `inputs` in the dynamic equations and  predictors.
Importantly, the names of the `predictors` should match the names of the signals in the dataset in order to construct the training and validation datasets as these signals will be used as the targets for the predictors.
The actual names of the dynamic states, on the other hand, has less importance, but should be unique within the configuration file.
However, if a predicted state is used as an input to another dynamic equation or predictor, the name of the state should also appear in the `states` list of the corresponding dynamic equation or predictor.

#### Latent Dynamics
There is also one additional residual example configuration file provided, `latent.yml`, which is used to specify a (black-box) latent residual configuration.
Instead of specifying unique dynamic equations, the latent residual configuration file specifies the number of latent dynamic states to be used, and the dataset signals to be used to infer the (latent) dynamics.
This should be used in conjunction with the optional `vae` parameter in the network configuration file to re-create the settings used in the ICML 2024 paper.
