# HyperLSTM

A [`flax`](https://github.com/google/flax) implementation of hyperlstms, based on [the original paper](https://arxiv.org/pdf/1609.09106.pdf).

### Setup

Most dependencies are installed with [poetry](https://python-poetry.org/). However, it makes more sense to install flax and jax using their instructions.
See the [jax](https://github.com/google/jax#installation) page for specific install instructions, for instance:  
`poetry run pip install --upgrade pip`

##### CPU  
`poetry run pip install --upgrade "jax[cpu]"`

##### GPU  
`poetry run pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html`  
Or, to install for a specific cuda version:  
`poetry run pip install --upgrade "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_releases.html`

##### TPU  
`poetry run pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"`

##### Tensorflow Text
Because of poor depedency specification, `tensorflow_text` must be installed via pip:
`poetry run pip install tensorflow_text`

Once jax is installed, install flax:  
`poetry run pip install flax`

##### Requirements
* TensorFlow dataset `glue/sst2` will be downloaded and prepared automatically, if necessary.

```bash
./run --workdir=/tmp/sst2 --config=configs/default.py`
```

#### Overriding Hyperparameter configurations

The SST2 example allows specifying a hyperparameter configuration by means of
setting the `--config` flag. The configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
./run \
    --workdir=/tmp/sst2 --config=configs/default.py \
    --config.learning_rate=0.05 --config.num_epochs=5
```
