# Copyright 2024
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

"""KAN implementation of entropy models for quantized latents."""

from typing import Any, Optional, Sequence, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np

from model import entropy_models
from model import laplace
from model import layers as layers_lib
from model import model_coding
from model.kan import KANConv2D, KANConv3D, SplineActivation

Array = chex.Array


class KANAutoregressiveEntropyModelConvVideo(
    hk.Module, model_coding.QuantizableMixin
):
  """Convolutional autoregressive entropy model for video using KAN.
  
  This replaces the MLPs in the entropy model with KAN implementations.
  """

  def __init__(
      self,
      num_grids: int,
      conditional_spec: config_dict.ConfigDict,
      mask_config: config_dict.ConfigDict,
      layers: Tuple[int, ...] = (12, 12),
      context_num_rows_cols: Union[int, Tuple[int, ...]] = 2,
      shift_log_scale: float = 0.0,
      scale_range: Optional[Tuple[float, float]] = None,
      clip_like_cool_chic: bool = True,
      use_linear_w_init: bool = True,
      num_knots: int = 15,  # KAN-specific parameter
      spline_range: float = 3.0,  # KAN-specific parameter
  ):
    """Constructor.

    Args:
      num_grids: Number of latent grids.
      conditional_spec: Spec determining the type of conditioning to apply.
      mask_config: mask_config used for entropy model.
      layers: Sizes of layers in the conv-net.
      context_num_rows_cols: Number of rows and columns to use as context for
        autoregressive prediction.
      shift_log_scale: Shift the `log_scale` by this amount before it is clipped
        and exponentiated.
      scale_range: Allowed range for scale of Laplace distribution.
      clip_like_cool_chic: If True, clips scale in Laplace distribution like in COOL-CHIC.
      use_linear_w_init: Whether to initialise the convolutions as if they were an MLP.
      num_knots: Number of knots for KAN spline approximation.
      spline_range: Range for KAN spline approximation.
    """
    super().__init__()
    # Need at least two layers
    assert len(layers) > 1, 'Need to have at least two layers.'
    self._scale_range = scale_range
    self._clip_like_cool_chic = clip_like_cool_chic
    self._num_grids = num_grids
    self._conditional_spec = conditional_spec
    self._mask_config = mask_config
    self._layers = layers
    self._shift_log_scale = shift_log_scale
    self._ndims = 3  # Video model
    self._num_knots = num_knots
    self._spline_range = spline_range

    if isinstance(context_num_rows_cols, tuple):
      assert len(context_num_rows_cols) == self._ndims
      self.context_num_rows_cols = context_num_rows_cols
    else:
      self.context_num_rows_cols = (context_num_rows_cols,) * self._ndims
    self.in_kernel_shape = tuple(2 * k + 1 for k in self.context_num_rows_cols)

    # Initialize the network similar to original but with KAN components
    self._initialize_network(use_linear_w_init)

  def _initialize_network(self, use_linear_w_init: bool):
    """Initialize the KAN network with masked convolutions."""
    mask, w_init = self._get_first_layer_mask_and_init(use_linear_w_init)
    
    # Build the network with KAN layers
    net = []
    
    # First layer is always a masked convolution
    net += [
        hk.Conv3D(
            output_channels=self._layers[0],
            kernel_shape=self.in_kernel_shape,
            mask=mask,
            w_init=w_init,
            name='masked_layer_0',
        ),
    ]
    
    # Intermediate layers use KANConv3D
    for i, width in enumerate(self._layers[1:]):
      net += [
          KANConv3D(
              output_channels=width,
              kernel_shape=1,  # 1x1x1 convolution
              num_knots=self._num_knots,
              spline_range=self._spline_range,
              name=f'kan_layer_{i+1}',
          ),
      ]
    
    # Final layer outputs 2 channels (loc, log_scale)
    net += [
        hk.Conv3D(
            output_channels=2,  # Output loc and log_scale
            kernel_shape=1,
            name='final_layer',
        ),
    ]
    
    self.net = hk.Sequential(net)

  def _get_first_layer_mask_and_init(
      self, use_linear_w_init: bool
  ) -> Tuple[Array, Optional[Any]]:
    """Returns the mask and weight initialization for the first layer.
    
    This replicates the functionality from the original entropy model.
    """
    if self._conditional_spec.use_conditioning:
      if self._conditional_spec.use_prev_grid:
        mask = layers_lib.get_prev_current_mask_3d(
            kernel_shape=self.in_kernel_shape,
            prev_kernel_shape=self._conditional_spec.prev_kernel_shape,
            f_out=self._layers[0],
        )
        w_init = None
      else:
        mask = layers_lib.causal_mask_3d(
            kernel_shape=self.in_kernel_shape,
            f_out=self._layers[0],
        )
        w_init = layers_lib.init_like_linear if use_linear_w_init else None
    else:
      mask = layers_lib.causal_mask_3d(
          kernel_shape=self.in_kernel_shape,
          f_out=self._layers[0],
      )
      w_init = layers_lib.init_like_linear if use_linear_w_init else None
    
    # Account for custom masking
    if self._mask_config.use_custom_masking:
      if self._mask_config.contiguous_mask_shape is not None:
        mask = layers_lib.get_contiguous_mask_3d(
            kernel_shape=self.in_kernel_shape,
            contiguous_shape=self._mask_config.contiguous_mask_shape,
            f_out=self._layers[0],
        )
      if self._mask_config.prev_frame_contiguous_mask_shape is not None:
        mask = layers_lib.get_prev_frame_contiguous_mask_3d(
            kernel_shape=self.in_kernel_shape,
            contiguous_shape=self._mask_config.prev_frame_contiguous_mask_shape,
            f_out=self._layers[0],
        )
    return mask, w_init

  def __call__(
      self,
      x_q_grids: Tuple[Array, ...],
      prev_x_q_grids: Optional[Tuple[Array, ...]] = None,
      is_training: bool = False,
  ) -> Array:
    """Calculate entropy model outputs for video data.
    
    Args:
      x_q_grids: Tuple of quantized latent grids.
      prev_x_q_grids: Optional tuple of previous frame's quantized latent grids.
      is_training: Whether the model is training.
      
    Returns:
      Rate (bits) required to encode the quantized latents.
    """
    # Follow same process as the original entropy model but with KAN network
    return entropy_models.AutoregressiveEntropyModelConvVideo._call_impl(
        self,
        x_q_grids=x_q_grids,
        prev_x_q_grids=prev_x_q_grids,
        is_training=is_training,
    )
  
  def _clip_log_scale(self, log_scale: Array) -> Array:
    """Clips log scale to lie in scale_range."""
    return entropy_models._clip_log_scale(
        log_scale, 
        self._scale_range, 
        self._clip_like_cool_chic
    )
