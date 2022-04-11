# HyperLSTM

A [`flax`](https://github.com/google/flax) implementation of hyperlstms, based on [the original paper](https://arxiv.org/pdf/1609.09106.pdf).

---

## Setup

Most dependencies are installed with [poetry](https://python-poetry.org/):

```
pip install poetry
poetry run pip install --upgrade pip
poetry install
```

For flax and jax, we have to install using their instructions.

### Step 1: Install jax for CPU, GPU, or TPU

You can choose to install jax for CPU, GPU, or TPU. Note, however, that we have not verified the TPU version on Windows, and it might lead to problems when trying to install a compatible jaxlib wheel in step 3.

For more information, see the [jax](https://github.com/google/jax#installation) page for specific install instructions.

At the time of this writing, on Windows machines, it is expected that installing jax will result in the following message similar to the one below, indicating jaxlib is not installed, but that jax has been installed. For instance, when running

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
optax 0.1.1 requires jaxlib>=0.1.37, which is not installed.
chex 0.1.2 requires jaxlib>=0.1.37, which is not installed.
Successfully installed jax-0.2.22
```

#### CPU

```bash
poetry run pip install --upgrade "jax[cpu]"
```

#### GPU

See [here](https://github.com/google/jax/blob/jax-v0.2.22/README.md#pip-installation-gpu-cuda) for more information.

```bash
poetry run pip install --upgrade "jax[cuda]" \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Or, to install for a specific cuda version, try something like:

```bash
poetry run pip install --upgrade "jax[cuda11_cudnn82]" \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

or:

```bash
poetry run pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

#### TPU

```bash
poetry run pip install "jax[tpu]>=0.2.16" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
```

### Step 2: Install jaxlib

For Linux users, installing jax may have successfully install jaxlib.

For Windows, pick a wheel from https://whls.blob.core.windows.net/unstable/index.html, and install it using the download link. The version of your wheel should be greater than or equal to the minor version of jax (e.g., at the time of this writing, jax-0.3.5 requires jaxlib version >= 0.3.0).

For more information on this, see [instructions for Windows users](https://github.com/cloudhan/jax-windows-builder).

Depending on whether your jax installation is for the CPU or for CUDA, you can select the corresponding CPU or CUDA wheel for jaxlib. For example, for CUDA installs:

```bash
poetry run pip install https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.5+cuda11.cudnn82-cp39-none-win_amd64.whl
```

### Step 3: Install flax

Flax will install easy now that we have first installed jax and jaxlib:

```bash
poetry run pip install flax
```

---

### Additional setup: Tensorflow Text

Because of poor depedency specification, `tensorflow_text` must be installed via pip:
```bash
poetry run pip install tensorflow_text
```

---

#### Requirements

- TensorFlow dataset `glue/sst2` will be downloaded and prepared automatically, if necessary.

```bash
./run --workdir=/tmp/sst2 --config=models/hyperlstm/configs/default.py
```

### Overriding Hyperparameter configurations

The SST2 example allows specifying a hyperparameter configuration by means of
setting the `--config` flag. The configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
./run \
    --workdir=/tmp/sst2 --config=models/hyperlstm/configs/default.py \
    --config.learning_rate=0.05 --config.num_epochs=5
```
