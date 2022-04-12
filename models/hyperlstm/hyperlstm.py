import abc
from functools import partial   # pylint: disable=g-importing-member
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from flax.linen.activation import sigmoid
from flax.linen.activation import tanh
from flax.linen.initializers import orthogonal
from flax.linen.initializers import zeros
from flax.linen.linear import Conv
from flax.linen.linear import default_kernel_init
from flax.linen.linear import Dense
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import Module
from flax.linen.reccurrent import *
from jax import numpy as jnp
from jax import random
import numpy as np

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any

class HyperLSTMCell(RNNCellBase):
  r""" A simple HyperLSTMCell
  The mathematical definition of the cell is the same as `LSTMCell` and as
  follows
  .. math::
      \begin{array}{ll}
      i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
      f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
      g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
      o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
      c' = f * c + i * g \\
      h' = o * \tanh(c') \\
      \end{array}
  where x is the input, h is the output of the previous time step, and c is
  the memory.

  However, instead of learning weights directly, each weight matrix is calculated as follows:
  .. math::
      \begin{array}{ll}
      i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
      f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
      g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
      o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
      c' = f * c + i * g \\
      h' = o * \tanh(c') \\
      \end{array}


  Attributes:
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: orthogonal).
    bias_init: initializer for the bias parameters (default: zeros).
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = orthogonal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32

  @compact
  def __call__(self, carry: Tuple[Array, Array],
               inputs: Array) -> Tuple[Tuple[Array, Array], Array]:
    r"""An optimized long short-term memory (LSTM) cell.
    Args:
      carry: the hidden state of the LSTM cell, initialized using
        `LSTMCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step. All
        dimensions except the final are considered batch dimensions.
    Returns:
      A tuple with the new carry and the output.
    """
    c, h = carry
    hidden_features = h.shape[-1]
    inputs = jnp.asarray(inputs, self.dtype)

    def _concat_dense(inputs: Array,
                      params: Mapping[str, Tuple[Array, Array]],
                      use_bias: bool = True) -> Array:
      # Concatenates the individual kernels and biases, given in params, into a
      # single kernel and single bias for efficiency before applying them using
      # dot_general.
      kernels, biases = zip(*params.values())
      kernel = jnp.asarray(jnp.concatenate(kernels, axis=-1), self.dtype)

      y = jnp.dot(inputs, kernel)
      if use_bias:
        bias = jnp.asarray(jnp.concatenate(biases, axis=-1), self.dtype)
        y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

      # Split the result back into individual (i, f, g, o) outputs.
      split_indices = np.cumsum([b.shape[0] for b in biases[:-1]])
      ys = jnp.split(y, split_indices, axis=-1)
      return dict(zip(params.keys(), ys))

    # Create params with the same names/shapes as `LSTMCell` for compatibility.
    dense_params_h = {}
    dense_params_i = {}
    for component in ['i', 'f', 'g', 'o']:
      dense_params_i[component] = DenseParams(
          features=hidden_features, use_bias=False,
          param_dtype=self.param_dtype,
          kernel_init=self.kernel_init, bias_init=self.bias_init,
          name=f'i{component}')(inputs)
      dense_params_h[component] = DenseParams(
          features=hidden_features, use_bias=True,
          param_dtype=self.param_dtype,
          kernel_init=self.recurrent_kernel_init, bias_init=self.bias_init,
          name=f'h{component}')(h)
    dense_h = _concat_dense(h, dense_params_h, use_bias=True)
    dense_i = _concat_dense(inputs, dense_params_i, use_bias=False)

    i = self.gate_fn(dense_h['i'] + dense_i['i'])
    f = self.gate_fn(dense_h['f'] + dense_i['f'])
    g = self.activation_fn(dense_h['g'] + dense_i['g'])
    o = self.gate_fn(dense_h['o'] + dense_i['o'])

    new_c = f * c + i * g
    new_h = o * self.activation_fn(new_c)
    return (new_c, new_h), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=zeros):
    """Initialize the RNN cell carry.
    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    key1, key2 = random.split(rng)
    mem_shape = batch_dims + (size,)
    return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


class HyperLSTM(nn.Module):
  """A simple hyper LSTM :)"""

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    return nn.OptimizedLSTMCell()(carry, x)

  @staticmethod
  def initialize_carry(batch_dims, hidden_size):
    # Use fixed random key since default state init fn is just zeros.
    return nn.OptimizedLSTMCell.initialize_carry(
        jax.random.PRNGKey(0), batch_dims, hidden_size)
