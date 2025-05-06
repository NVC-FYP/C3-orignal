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

"""Kolmogorov-Arnold Networks (KAN) implementation for C3."""

from typing import Optional, Callable, Any, Sequence, Union
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class SplineActivation(hk.Module):
    """Learnable spline activation function for KAN."""
    
    def __init__(
        self,
        num_knots: int = 10,
        spline_range: float = 3.0,
        init_scheme: str = "relu_like",
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.num_knots = num_knots
        self.spline_range = spline_range
        self.init_scheme = init_scheme
        
    def __call__(self, x):
        # Initialize knots uniformly between -spline_range and spline_range
        knot_xs = jnp.linspace(-self.spline_range, self.spline_range, self.num_knots)
        
        # Initialize knot values (heights) based on init_scheme
        if self.init_scheme == "relu_like":
            # Initialize to approximate ReLU
            init_ys = jnp.where(knot_xs > 0, knot_xs, 0)
            init_fn = lambda shape, dtype: jnp.array(init_ys, dtype=dtype)
        else:
            # Random initialization
            init_fn = hk.initializers.RandomNormal(stddev=0.01)
        
        knot_ys = hk.get_parameter("knot_ys", shape=[self.num_knots], init=init_fn)
        
        # Linear interpolation
        x_clipped = jnp.clip(x, -self.spline_range, self.spline_range)
        
        # Find which knot interval each x falls into
        idx = jnp.searchsorted(knot_xs, x_clipped) - 1
        idx = jnp.clip(idx, 0, self.num_knots - 2)
        
        # Get surrounding knot positions and values
        x0 = knot_xs[idx]
        x1 = knot_xs[idx + 1]
        y0 = jnp.take(knot_ys, idx)
        y1 = jnp.take(knot_ys, idx + 1)
        
        # Linear interpolation formula: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        slope = (y1 - y0) / (x1 - x0)
        y = y0 + slope * (x_clipped - x0)
        
        return y


class KANConv2D(hk.Module):
    """2D Convolutional layer with KAN spline activations."""
    
    def __init__(
        self,
        output_channels: int,
        kernel_shape: Union[int, Sequence[int]],
        num_knots: int = 10,
        spline_range: float = 3.0,
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.num_knots = num_knots
        self.spline_range = spline_range
        self.stride = stride
        self.rate = rate
        self.padding = padding
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init
        
    def __call__(self, x):
        # Apply regular convolution
        conv = hk.Conv2D(
            output_channels=self.output_channels,
            kernel_shape=self.kernel_shape,
            stride=self.stride,
            rate=self.rate,
            padding=self.padding,
            with_bias=self.with_bias,
            w_init=self.w_init,
            b_init=self.b_init,
        )(x)
        
        # Apply learnable spline activation
        return SplineActivation(num_knots=self.num_knots, spline_range=self.spline_range)(conv)


class KANConv3D(hk.Module):
    """3D Convolutional layer with KAN spline activations."""
    
    def __init__(
        self,
        output_channels: int,
        kernel_shape: Union[int, Sequence[int]],
        num_knots: int = 10,
        spline_range: float = 3.0,
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.num_knots = num_knots
        self.spline_range = spline_range
        self.stride = stride
        self.rate = rate
        self.padding = padding
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init
        
    def __call__(self, x):
        # Apply regular convolution
        conv = hk.Conv3D(
            output_channels=self.output_channels,
            kernel_shape=self.kernel_shape,
            stride=self.stride,
            rate=self.rate,
            padding=self.padding,
            with_bias=self.with_bias,
            w_init=self.w_init,
            b_init=self.b_init,
        )(x)
        
        # Apply learnable spline activation
        return SplineActivation(num_knots=self.num_knots, spline_range=self.spline_range)(conv)