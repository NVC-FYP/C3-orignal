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

"""Implementation of Synthesis network using KAN (Kolmogorov-Arnold Networks)."""

import functools
from typing import Any, Optional, Sequence, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from model import model_coding
from model import synthesis
from model.kan import KANConv2D, KANConv3D, SplineActivation

Array = jnp.ndarray


class KANSynthesis(hk.Module, model_coding.QuantizableMixin):
  """Synthesis network using KAN: Kolmogorov-Arnold Networks in place of MLPs."""

  def __init__(
      self,
      *,
      layers: Tuple[int, ...] = (12, 12),
      out_channels: int = 3,
      kernel_shape: int = 1,
      num_residual_layers: int = 0,
      residual_kernel_shape: int = 3,
      add_activation_before_residual: bool = False,
      add_layer_norm: bool = False,
      clip_range: Tuple[float, float] = (0.0, 1.0),
      is_video: bool = False,
      per_frame_conv: bool = False,
      b_last_init_value: Optional[Array] = None,
      num_knots: int = 15,  # KAN-specific parameter
      spline_range: float = 3.0,  # KAN-specific parameter
      **unused_kwargs,
  ):
    """Constructor.

    Args:
      layers: Sequence of layer sizes. Length of tuple corresponds to depth of
        network.
      out_channels: Number of output channels.
      kernel_shape: Shape of convolutional kernel.
      num_residual_layers: Number of extra residual conv layers.
      residual_kernel_shape: Kernel shape of extra residual conv layers.
      add_activation_before_residual: If True, adds a nonlinearity before the
        residual layers.
      add_layer_norm: Whether to add layer norm to the input.
      clip_range: Range at which outputs will be clipped.
      is_video: If True, synthesizes a video, otherwise synthesizes an image.
      per_frame_conv: If True, applies 2D residual convolution layers *per*
        frame. If False, applies 3D residual convolutional layer directly to 3D
        volume of latents. Only used when is_video is True.
      b_last_init_value: Optional. Array to be used as initial setting for the
        bias in the last layer (residual or non-residual) of the network.
      num_knots: Number of knots for KAN spline approximation.
      spline_range: Range for KAN spline approximation.
    """
    super().__init__()
    self._output_clip_range = clip_range
    
    b_last_init = lambda shape, dtype: synthesis.b_init_custom_value(
        shape, dtype, b_last_init_value)

    # Initialize layers with KAN components
    net_layers = []
    
    # Select appropriate KAN and regular convolution constructors based on input type
    if is_video:
      kan_conv = KANConv3D
      regular_conv = hk.Conv3D
    else:
      kan_conv = KANConv2D
      regular_conv = hk.Conv2D

    if add_layer_norm:
      net_layers += [
          hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
      ]
      
    # Replace standard MLP layers (conv + activation) with KAN layers
    for layer_size in layers:
      net_layers += [
          kan_conv(
              output_channels=layer_size,
              kernel_shape=kernel_shape,
              num_knots=num_knots,
              spline_range=spline_range,
          ),
      ]
    
    # Final layer without activation
    net_layers += [
        regular_conv(
            out_channels,
            kernel_shape=kernel_shape,
            b_init=None if num_residual_layers > 0 else b_last_init,
        )
    ]

    # Optionally add activation before residual layers
    if num_residual_layers > 0 and add_activation_before_residual:
      spline_activation = SplineActivation(
          num_knots=num_knots, 
          spline_range=spline_range,
      )
      net_layers += [spline_activation]
    
    # Decide convolution types for residual layers
    if is_video and not per_frame_conv:
      residual_kan_conv = KANConv3D
      residual_conv = hk.Conv3D
    else:
      residual_kan_conv = KANConv2D
      residual_conv = hk.Conv2D
    
    # Add residual layers (based on the original implementation)
    for i in range(num_residual_layers):
      is_last_layer = i == num_residual_layers - 1
      
      # Use KAN for all but the last layer, where we use regular conv with custom bias
      if is_last_layer:
        core_conv = residual_conv(
            out_channels,
            kernel_shape=residual_kernel_shape,
            padding='VALID',
            w_init=jnp.zeros,
            b_init=b_last_init if is_last_layer else None,
            name='residual_conv',
        )
      else:
        core_conv = residual_kan_conv(
            output_channels=out_channels,
            kernel_shape=residual_kernel_shape,
            padding='VALID',
            w_init=jnp.zeros,
            num_knots=num_knots,
            spline_range=spline_range,
            name='residual_kan_conv',
        )
      
      net_layers += [
          synthesis.ResidualWrapper(
              hk.Sequential([
                  # Add edge padding for well-behaved synthesis at boundaries
                  functools.partial(
                      synthesis.edge_padding,
                      kernel_shape=residual_kernel_shape,
                      is_video=is_video,
                      per_frame_conv=per_frame_conv,
                  ),
                  core_conv,
              ]),
          )
      ]
      
      # Add activation for all but the last residual layer
      if not is_last_layer:
        net_layers += [
            SplineActivation(
                num_knots=num_knots,
                spline_range=spline_range,
            )
        ]

    self._net = hk.Sequential(net_layers)

  def __call__(self, latents: Array) -> Array:
    """Maps latents to image or video using KAN.

    Args:
      latents: Array of latents of shape ({T}, H, W, C).

    Returns:
      Predicted image or video of shape ({T}, H, W, out_channels).
    """
    return jnp.clip(
        self._net(latents),
        self._output_clip_range[0],
        self._output_clip_range[1],
    )
