# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Entropy models for quantized latents."""

from collections.abc import Callable
from typing import Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np

from model import laplace
from model import layers as layers_lib
from model import model_coding

# Try importing KAN, handle potential import error
try:
  from kan import KAN
except ImportError:
  KAN = None
  print("WARNING: pykan library not found. KAN functionality will be disabled.")


Array = chex.Array
init_like_linear = layers_lib.init_like_linear
causal_mask = layers_lib.causal_mask


def _clip_log_scale(
    log_scale: Array,
    scale_range: tuple[float, float],
    clip_like_cool_chic: bool = True,
) -> Array:
  """Clips log scale to lie in `scale_range`."""
  if clip_like_cool_chic:
    # This slightly odd clipping is based on the COOL-CHIC implementation
    # https://github.com/Orange-OpenSource/Cool-Chic/blob/16c41c033d6fd03e9f038d4f37d1ca330d5f7e35/src/models/arm.py#L158
    log_scale = -0.5 * jnp.clip(
        log_scale,
        -2 * jnp.log(scale_range[1]),
        -2 * jnp.log(scale_range[0]),
    )
  else:
    log_scale = jnp.clip(
        log_scale,
        jnp.log(scale_range[0]),
        jnp.log(scale_range[1]),
    )
  return log_scale


class AutoregressiveEntropyModelConvImage(
    hk.Module, model_coding.QuantizableMixin
):
  """Convolutional autoregressive entropy model for COOL-CHIC. Image only.

  This convolutional version is mathematically equivalent to its non-
  convolutional counterpart but also supports explicit batch dimensions.
  Can optionally use KAN instead of MLP (1x1 Convs).
  """

  def __init__(
      self,
      conditional_spec: config_dict.ConfigDict,
      layers: tuple[int, ...] = (12, 12),
      activation_fn: str = 'gelu',
      context_num_rows_cols: int | tuple[int, int] = 2,
      shift_log_scale: float = 0.0,
      scale_range: tuple[float, float] | None = None,
      clip_like_cool_chic: bool = True,
      use_linear_w_init: bool = True,
      # KAN specific config (passed from main config)
      kan_config: config_dict.ConfigDict | None = None,
  ):
    """Constructor.

    Args:
      conditional_spec: Spec determining the type of conditioning to apply.
      layers: Sizes of hidden layers in the MLP/KAN. The final output size is 2.
      activation_fn: Activation function of conv net (used if not KAN).
      context_num_rows_cols: Number of rows and columns to use as context for
        autoregressive prediction. Can be an integer, in which case the number
        of rows and columns is equal, or a tuple. The kernel size of the first
        convolution is given by `2*context_num_rows_cols + 1` (in each
        dimension).
      shift_log_scale: Shift the `log_scale` by this amount before it is clipped
        and exponentiated.
      scale_range: Allowed range for scale of Laplace distribution. For example,
        if scale_range = (1.0, 2.0), the scales are clipped to lie in [1.0,
        2.0]. If `None` no clipping is applied.
      clip_like_cool_chic: If True, clips scale in Laplace distribution in the
        same way as it's done in COOL-CHIC codebase. This involves clipping a
        transformed version of the log scale.
      use_linear_w_init: Whether to initialise the convolutions as if they were
        an MLP.
      kan_config: ConfigDict containing KAN settings ('use_kan', 'grid_size',
        'spline_order').
    """
    super().__init__(name='autoregressive_entropy_model_conv_image') # Ensure consistent naming
    self._layers = layers
    self._activation_fn = getattr(jax.nn, activation_fn)
    self._scale_range = scale_range
    self._clip_like_cool_chic = clip_like_cool_chic
    self._conditional_spec = conditional_spec
    self._shift_log_scale = shift_log_scale
    self._use_linear_w_init = use_linear_w_init
    self._kan_config = kan_config if kan_config is not None else config_dict.ConfigDict({'use_kan': False})
    self._use_kan = self._kan_config.get('use_kan', False)

    if self._use_kan and KAN is None:
        raise ImportError("KAN requires the 'pykan' library to be installed.")

    if isinstance(context_num_rows_cols, tuple):
      self.context_num_rows_cols = context_num_rows_cols
    else:
      self.context_num_rows_cols = (context_num_rows_cols,) * 2
    self.in_kernel_shape = tuple(2 * k + 1 for k in self.context_num_rows_cols)

    mask, w_init = self._get_first_layer_mask_and_init()

    # --- Input Layer (Masked Conv) ---
    self.masked_conv = hk.Conv2D(
        output_channels=layers[0],
        kernel_shape=self.in_kernel_shape,
        mask=mask,
        w_init=w_init,
        name='masked_layer_0',
    )

    # --- Processing Layers (MLP or KAN) ---
    if self._use_kan:
      # Define KAN layer
      kan_width = [layers[0]] + list(layers[1:]) + [layers[-1]] # KAN processes features up to the dimension before final output projection
      self.kan_processor_core = KAN(
          width=kan_width,
          grid=self._kan_config.grid_size,
          k=self._kan_config.spline_order,
          name='kan_processor'
      )
      # Use BatchApply to apply KAN pixel-wise
      self.kan_processor = hk.BatchApply(self.kan_processor_core)
      # Define the final projection layer separately
      self.final_projection = hk.Conv2D(
          output_channels=2, # loc and scale
          kernel_shape=1,
          name='final_projection'
      )
    else:
      # Define MLP using 1x1 Convs
      net = []
      for i, width in enumerate(layers[1:]):
        net += [
            self._activation_fn,
            hk.Conv2D(
                output_channels=width,
                kernel_shape=1,
                name=f'layer_{i+1}',
            ),
        ]
      # Final layer producing loc and scale
      net += [
          self._activation_fn,
          hk.Conv2D(
              output_channels=2,
              kernel_shape=1,
              name=f'layer_{len(layers)}',
          ),
      ]
      self.mlp_processor = hk.Sequential(net)

  def _get_first_layer_mask_and_init(
      self,
  ) -> tuple[Array, Callable[[Any, Any], Array] | None]:
    """Returns the mask and weight initialization of the first layer."""
    if self._conditional_spec.use_conditioning:
      if self._conditional_spec.use_prev_grid:
        mask = layers_lib.get_prev_current_mask(
            kernel_shape=self.in_kernel_shape,
            prev_kernel_shape=self._conditional_spec.prev_kernel_shape,
            f_out=self._layers[0],
        )
        w_init = None # Use default Haiku init for conditional conv
      else:
        raise ValueError('Only use_prev_grid conditioning supported.')
    else:
      mask = causal_mask(
          kernel_shape=self.in_kernel_shape, f_out=self._layers[0]
      )
      w_init = init_like_linear if self._use_linear_w_init else None
    return mask, w_init

  def _get_mask(
      self,
      dictkey: tuple[jax.tree_util.DictKey, ...],  # From `tree_map_with_path`.
      array: Array,
  ) -> Array:
    # Apply mask only to the first convolutional layer if not using KAN
    # or if using conditioning. KAN layers handle their internal structure.
    # Final projection layer (1x1 Conv) doesn't need masking.
    assert isinstance(dictkey[0].key, str)
    if 'masked_layer_0/w' in dictkey[0].key:
        mask, _ = self._get_first_layer_mask_and_init()
        # Handle conditional input channel dimension if necessary
        if mask.shape[-2] != array.shape[-2]: # Check f_in dimension
             mask = np.broadcast_to(mask, array.shape)
        return mask
    elif self._use_kan and 'kan_processor' in dictkey[0].key:
        # Let KAN handle its parameters internally, assume no masking needed here
        # Or potentially implement masking within KAN if required, but pykan might not support it directly.
        return np.ones(shape=array.shape, dtype=bool)
    elif not self._use_kan and 'mlp_processor' in dictkey[0].key:
        # MLP layers (1x1 conv) don't need explicit masking beyond the first layer
        return np.ones(shape=array.shape, dtype=bool)
    elif 'final_projection' in dictkey[0].key:
         # Final 1x1 conv layer doesn't need masking
        return np.ones(shape=array.shape, dtype=bool)
    else:
      # Default for biases or other potential params
      return np.ones(shape=array.shape, dtype=bool)

  def __call__(self, latent_grids: tuple[Array, ...]) -> tuple[Array, Array]:
    """Maps latent grids to parameters of Laplace distribution for every latent.

    Args:
      latent_grids: Tuple of all latent grids of shape (H, W), (H/2, W/2), etc.
                      or (B, H, W), (B, H/2, W/2), etc.

    Returns:
      Tuple of parameters of Laplace distribution (loc and scale) each of shape
      (num_latents,) or (B, num_latents).
    """
    has_batch_dim = len(latent_grids[0].shape) == 3
    bs = latent_grids[0].shape[0] if has_batch_dim else None

    if self._conditional_spec.use_conditioning:
      if self._conditional_spec.use_prev_grid:
        grids_cond = (jnp.zeros_like(latent_grids[0]),) + latent_grids[:-1]
        dist_params = []
        for prev_grid, grid in zip(grids_cond, latent_grids):
          # Resize `prev_grid` to have the same resolution as the current grid
          prev_grid_resized = jax.image.resize(
              prev_grid,
              shape=grid.shape[1:] if has_batch_dim else grid.shape, # Resize spatial dims only
              method=self._conditional_spec.interpolation
          )
          # Add batch dim back if it was removed
          if has_batch_dim and prev_grid_resized.ndim == 2:
              prev_grid_resized = jnp.expand_dims(prev_grid_resized, axis=0)

          inputs = jnp.stack(
              [prev_grid_resized, grid], axis=-1
          ) # (B, H, W, 2) or (H, W, 2)

          # Apply masked conv and activation
          context = self.masked_conv(inputs)
          context = self._activation_fn(context)

          # Apply KAN or MLP
          if self._use_kan:
              processed = self.kan_processor(context)
              out = self.final_projection(processed)
          else:
              out = self.mlp_processor(context)
          dist_params.append(out)
      else:
        raise ValueError('use_prev_grid is False, but conditioning requested')
    else:
      # Apply network to each grid without conditioning on previous grid
      dist_params = []
      for grid in latent_grids:
          # Add channel dim if missing
          if grid.ndim == 2: # (H, W)
              current_input = grid[..., None]
          elif grid.ndim == 3 and has_batch_dim: # (B, H, W)
              current_input = grid[..., None]
          else: # (H, W, C) or (B, H, W, C) - should be C=1
              assert grid.shape[-1] == 1
              current_input = grid

          # Apply masked conv and activation
          context = self.masked_conv(current_input)
          context = self._activation_fn(context)

          # Apply KAN or MLP
          if self._use_kan:
              processed = self.kan_processor(context)
              out = self.final_projection(processed)
          else:
              out = self.mlp_processor(context)
          dist_params.append(out)


    # Flatten results
    if bs is not None:
      # Reshape preserving batch dim: (B, H, W, 2) -> (B, H*W, 2)
      dist_params = [p.reshape(bs, -1, 2) for p in dist_params]
      dist_params = jnp.concatenate(dist_params, axis=1)  # (B, num_latents, 2)
    else:
      # Reshape removing spatial dims: (H, W, 2) -> (H*W, 2)
      dist_params = [p.reshape(-1, 2) for p in dist_params]
      dist_params = jnp.concatenate(dist_params, axis=0)  # (num_latents, 2)

    chex.assert_shape(dist_params, (bs, None, 2) if bs else (None, 2))
    loc, log_scale = dist_params[..., 0], dist_params[..., 1]

    log_scale = log_scale + self._shift_log_scale

    # Optionally clip log scale (we clip scale in log space to avoid overflow).
    if self._scale_range is not None:
      log_scale = _clip_log_scale(
          log_scale, self._scale_range, self._clip_like_cool_chic
      )
    # Convert log scale to scale (which ensures scale is positive)
    scale = jnp.exp(log_scale)
    return loc, scale

# --- Video Model ---
# (Similar structure, need to adapt __init__ and __call__)

class AutoregressiveEntropyModelConvVideo(
    hk.Module, model_coding.QuantizableMixin
):
  """Convolutional autoregressive entropy model for COOL-CHIC on video.
     Can optionally use KAN instead of MLP (1x1x1 Convs).
  """

  def __init__(
      self,
      num_grids: int,
      conditional_spec: config_dict.ConfigDict,
      mask_config: config_dict.ConfigDict,
      layers: tuple[int, ...] = (12, 12),
      activation_fn: str = 'gelu',
      context_num_rows_cols: int | tuple[int, ...] = 2,
      shift_log_scale: float = 0.0,
      scale_range: tuple[float, float] | None = None,
      clip_like_cool_chic: bool = True,
      use_linear_w_init: bool = True, # Not applicable for KAN part
      # KAN specific config (passed from main config)
      kan_config: config_dict.ConfigDict | None = None,
  ):
    """Constructor. (See Image model for args description)."""
    super().__init__(name='autoregressive_entropy_model_conv_video') # Ensure consistent naming
    assert len(layers) > 0, 'Need to have at least one layer size.'
    self._activation_fn = getattr(jax.nn, activation_fn)
    self._scale_range = scale_range
    self._clip_like_cool_chic = clip_like_cool_chic
    self._num_grids = num_grids
    self._conditional_spec = conditional_spec
    self._mask_config = mask_config
    self._layers = layers
    self._shift_log_scale = shift_log_scale
    self._ndims = 3  # Video model.
    self._kan_config = kan_config if kan_config is not None else config_dict.ConfigDict({'use_kan': False})
    self._use_kan = self._kan_config.get('use_kan', False)

    if self._use_kan and KAN is None:
        raise ImportError("KAN requires the 'pykan' library to be installed.")

    if isinstance(context_num_rows_cols, tuple):
      assert len(context_num_rows_cols) == self._ndims
      self.context_num_rows_cols = context_num_rows_cols
    else:
      self.context_num_rows_cols = (context_num_rows_cols,) * self._ndims
    self.in_kernel_shape = tuple(2 * k + 1 for k in self.context_num_rows_cols)


    # --- Define Network Builders ---
    def build_network_modules(prefix):
        modules = {}
        # Input Layer (Masked Conv or EfficientConv)
        if mask_config.use_custom_masking:
            assert self._conditional_spec.type == 'per_grid'
            current_frame_kw = mask_config.current_frame_mask_size
            assert current_frame_kw % 2 == 1
            current_frame_ks = (current_frame_kw, current_frame_kw)
            prev_frame_ks = mask_config.prev_frame_contiguous_mask_shape
            modules['input_conv'] = layers_lib.EfficientConv(
                output_channels=layers[0],
                kernel_shape_current=current_frame_ks,
                kernel_shape_prev=prev_frame_ks,
                kernel_shape_conv3d=self.in_kernel_shape,
                name=f'{prefix}input_efficient_conv',
            )
        else:
            mask = causal_mask(kernel_shape=self.in_kernel_shape, f_out=layers[0])
            modules['input_conv'] = hk.Conv3D(
                output_channels=layers[0],
                kernel_shape=self.in_kernel_shape,
                mask=mask,
                w_init=init_like_linear if use_linear_w_init else None,
                name=f'{prefix}input_masked_conv',
            )

        # Processing Layers (MLP or KAN)
        if self._use_kan:
            kan_width = [layers[0]] + list(layers[1:]) + [layers[-1]]
            kan_core = KAN(
                width=kan_width,
                grid=self._kan_config.grid_size,
                k=self._kan_config.spline_order,
                name=f'{prefix}kan_processor'
            )
            modules['processor'] = hk.BatchApply(kan_core) # Apply voxel-wise
            modules['final_projection'] = hk.Conv3D(
                output_channels=2, kernel_shape=1, name=f'{prefix}final_projection'
            )
        else:
            mlp_layers = []
            for i, width in enumerate(layers[1:]):
                 mlp_layers += [
                     self._activation_fn,
                     hk.Conv3D(
                         output_channels=width, kernel_shape=1, name=f'{prefix}layer_{i+1}'
                     ),
                 ]
            # Final layer producing loc and scale
            mlp_layers += [
                self._activation_fn,
                hk.Conv3D(
                    output_channels=2, kernel_shape=1, name=f'{prefix}layer_{len(layers)}'
                ),
            ]
            modules['processor'] = hk.Sequential(mlp_layers)
            modules['final_projection'] = None # Projection is part of the Sequential MLP

        return modules

    # --- Instantiate Networks ---
    if self._conditional_spec and self._conditional_spec.use_conditioning:
        assert self._conditional_spec.type == 'per_grid'
        self.nets_modules = [build_network_modules(f'grid_{i}_') for i in range(self._num_grids)]
    else:
        self.net_modules = build_network_modules('') # Single network for all grids

  def _get_mask(
      self,
      dictkey: tuple[jax.tree_util.DictKey, ...],  # From `tree_map_with_path`.
      array: Array,
  ) -> Array:
    # Handles masking for both MLP and KAN based models
    assert isinstance(dictkey[0].key, str)
    module_key = dictkey[0].key

    if 'input_masked_conv/w' in module_key:
        return causal_mask(kernel_shape=self.in_kernel_shape, f_out=self._layers[0])
    elif 'input_efficient_conv' in module_key:
         # EfficientConv handles its own masking internally via hk.Conv2D mask arg
         if 'conv_current_masked_layer/w' in module_key:
             kw = kh = self._mask_config.current_frame_mask_size
             return causal_mask(kernel_shape=(kh, kw), f_out=self._layers[0])
         else: # Biases or prev_conv weights in EfficientConv don't need external masking
             return np.ones(shape=array.shape, dtype=bool)
    elif (self._use_kan and 'kan_processor' in module_key) or \
         (not self._use_kan and 'processor' in module_key and 'layer_' in module_key) or \
         ('final_projection' in module_key):
        # KAN layers, subsequent MLP layers (1x1x1 Conv3D), and final projection don't need masking
        return np.ones(shape=array.shape, dtype=bool)
    else:
        # Default for biases or other potential params
        return np.ones(shape=array.shape, dtype=bool)

  def __call__(
      self,
      latent_grids: tuple[Array, ...],
      prev_frame_mask_top_lefts: tuple[tuple[int, int] | None, ...] | None = None,
  ) -> tuple[Array, Array]:
    """Maps latent grids to parameters of Laplace distribution for every latent.

    Args:
      latent_grids: Tuple of latent grids, shapes like (T, H, W), (T/2, H/2, W/2), ...
                      or (B, T, H, W), (B, T/2, H/2, W/2), ...
      prev_frame_mask_top_lefts: Used only when `mask_config.use_custom_masking=True`.
                                 Tuple of top-left coords for prev frame mask per grid.

    Returns:
      loc, scale: Parameters of Laplace distribution, shapes (num_latents,) or (B, num_latents).
    """
    assert len(latent_grids) == self._num_grids
    has_batch_dim = len(latent_grids[0].shape) == 4
    bs = latent_grids[0].shape[0] if has_batch_dim else None

    dist_params = []

    for k, grid in enumerate(latent_grids):
        # Select the appropriate network modules
        if self._conditional_spec and self._conditional_spec.use_conditioning:
            current_net_modules = self.nets_modules[k]
        else:
            current_net_modules = self.net_modules

        # Add channel dim if input doesn't have one
        if grid.ndim == 3: # (T, H, W)
             current_input = grid[..., None]
        elif grid.ndim == 4 and has_batch_dim: # (B, T, H, W)
             current_input = grid[..., None]
        else: # (T, H, W, C) or (B, T, H, W, C) - should be C=1
             assert grid.shape[-1] == 1
             current_input = grid

        # --- Apply Network ---
        # 1. Input Convolution (Masked Conv3D or EfficientConv)
        if self._mask_config.use_custom_masking:
             # EfficientConv takes current_input and mask_info
             context = current_net_modules['input_conv'](
                 current_input, prev_frame_mask_top_left=prev_frame_mask_top_lefts[k]
             )
        else:
             # Standard Masked Conv3D
             context = current_net_modules['input_conv'](current_input)

        # 2. Activation
        activated_context = self._activation_fn(context)

        # 3. Processor (KAN or MLP) and Final Projection
        if self._use_kan:
             processed = current_net_modules['processor'](activated_context)
             dp = current_net_modules['final_projection'](processed)
        else: # MLP
             dp = current_net_modules['processor'](activated_context)

        dist_params.append(dp)

    # Flatten results
    if bs is not None:
      # Reshape preserving batch dim: (B, T, H, W, 2) -> (B, T*H*W, 2)
      dist_params = [p.reshape(bs, -1, 2) for p in dist_params]
      dist_params = jnp.concatenate(dist_params, axis=1)  # (B, num_latents, 2)
    else:
      # Reshape removing spatial/temporal dims: (T, H, W, 2) -> (T*H*W, 2)
      dist_params = [p.reshape(-1, 2) for p in dist_params]
      dist_params = jnp.concatenate(dist_params, axis=0)  # (num_latents, 2)


    chex.assert_shape(dist_params, (bs, None, 2) if bs else (None, 2))
    loc, log_scale = dist_params[..., 0], dist_params[..., 1]

    log_scale = log_scale + self._shift_log_scale

    # Optionally clip log scale
    if self._scale_range is not None:
      log_scale = _clip_log_scale(
          log_scale, self._scale_range, self._clip_like_cool_chic
      )
    scale = jnp.exp(log_scale)
    return loc, scale


def compute_rate(
    x: Array, loc: Array, scale: Array, q_step: float = 1.0
) -> Array:
  """Compute entropy of x (in bits) under the Laplace(mu, scale) distribution.

  Args:
    x: Array of shape (...,) containing points whose entropy will be evaluated.
    loc: Array of shape (...,) containing the location (mu) parameter.
    scale: Array of shape (...,) containing scale parameter.
    q_step: Step size used for quantizing x.

  Returns:
    Rate (entropy) of x under model as array of same shape as x.
  """
  # Ensure computation happens in the space where bin width is 1.
  x /= q_step
  loc /= q_step
  scale /= q_step

  dist = laplace.Laplace(loc, scale)
  log_probs = dist.integrated_log_prob(x)

  # Change base of logarithm
  rate = - log_probs / jnp.log(2.)

  # Clip rate (COOL-CHIC implementation detail)
  rate = jnp.clip(rate, a_max=16)

  return rate


def flatten_latent_grids(latent_grids: tuple[Array, ...]) -> Array:
  """Flattens list of latent grids into a single array.

  Args:
    latent_grids: Tuple of grids, shapes like ({B}, {T}, H, W).

  Returns:
    Array of shape (num_latents,) or (B, num_latents).
  """
  has_batch_dim = latent_grids[0].ndim > 3 # Check if batch dim exists

  if has_batch_dim:
      bs = latent_grids[0].shape[0]
      # Reshape each grid from (B, {T}, H, W) to (B, {T}*H*W)
      all_latents = [grid.reshape(bs, -1) for grid in latent_grids]
      # Stack along feature dimension: (B, N_total)
      return jnp.concatenate(all_latents, axis=1)
  else:
      # Reshape each grid from ({T}, H, W) to ({T}*H*W,)
      all_latents = [grid.reshape(-1) for grid in latent_grids]
      # Stack into a single array: (N_total,)
      return jnp.concatenate(all_latents, axis=0)


def unflatten_latent_grids(
    flattened_latents: Array,
    latent_grid_shapes: tuple[tuple[int, ...], ...]
) -> tuple[Array, ...]:
  """Unflattens a single flattened latent array into a tuple of latent grids.

  Args:
    flattened_latents: Flattened latent grids (1D or 2D if batch dim exists).
    latent_grid_shapes: Tuple of shapes of latent grids (excluding batch dim).

  Returns:
    Tuple of latent grids with original shapes (potentially including batch dim).
  """
  has_batch_dim = flattened_latents.ndim == 2
  num_total_latents = sum([np.prod(s) for s in latent_grid_shapes])

  if has_batch_dim:
      bs = flattened_latents.shape[0]
      chex.assert_shape(flattened_latents, (bs, num_total_latents))
  else:
      bs = None
      chex.assert_shape(flattened_latents, (num_total_latents,))

  latent_grids = []
  current_idx = 0
  for shape in latent_grid_shapes:
    size = int(np.prod(shape)) # Size excluding batch dim
    if has_batch_dim:
        # Select columns for the current grid, reshape including batch dim
        grid = flattened_latents[:, current_idx : current_idx + size]
        grid = grid.reshape((bs,) + shape)
    else:
        # Select elements for the current grid, reshape
        grid = flattened_latents[current_idx : current_idx + size]
        grid = grid.reshape(shape)
    latent_grids.append(grid)
    current_idx += size

  return tuple(latent_grids)