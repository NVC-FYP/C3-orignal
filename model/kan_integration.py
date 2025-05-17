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

"""Utility functions to integrate KAN models into the C3 architecture."""

import functools
import copy
from typing import Any, Dict, Optional, Tuple

import jax
import haiku as hk

from model import entropy_models
from model import kan_entropy_models
from model import synthesis
from model import kan_synthesis


def get_synthesis_model(config, is_video=False):
    """Factory function to get either standard MLP synthesis or KAN synthesis.
    
    Args:
        config: Configuration object with model parameters
        is_video: Whether the model is for video data
        
    Returns:
        Appropriate synthesis module class
    """
    if hasattr(config.model, 'use_kan') and config.model.use_kan:
        # Use KAN-based synthesis network
        return functools.partial(
            kan_synthesis.KANSynthesis,
            layers=config.model.synthesis.layers,
            out_channels=3,  # RGB
            kernel_shape=config.model.synthesis.kernel_shape,
            num_residual_layers=config.model.synthesis.num_residual_layers,
            residual_kernel_shape=config.model.synthesis.residual_kernel_shape,
            add_activation_before_residual=config.model.synthesis.add_activation_before_residual,
            add_layer_norm=config.model.synthesis.add_layer_norm,
            is_video=is_video,
            per_frame_conv=config.model.synthesis.per_frame_conv,
            num_knots=config.model.kan.num_knots,
            spline_range=config.model.kan.spline_range,
        )
    else:
        # Use standard MLP-based synthesis network
        return functools.partial(
            synthesis.Synthesis,
            layers=config.model.synthesis.layers,
            out_channels=3,  # RGB
            kernel_shape=config.model.synthesis.kernel_shape,
            num_residual_layers=config.model.synthesis.num_residual_layers,
            residual_kernel_shape=config.model.synthesis.residual_kernel_shape,
            activation_fn=config.model.synthesis.activation_fn,
            add_activation_before_residual=config.model.synthesis.add_activation_before_residual,
            add_layer_norm=config.model.synthesis.add_layer_norm,
            is_video=is_video,
            per_frame_conv=config.model.synthesis.per_frame_conv,
        )


def get_entropy_model_video(config):
    """Factory function to get either standard MLP entropy model or KAN entropy model for video.
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        Appropriate entropy model
    """
    if hasattr(config.model, 'use_kan') and config.model.use_kan:
        # Use KAN-based entropy model
        return functools.partial(
            kan_entropy_models.KANAutoregressiveEntropyModelConvVideo,
            num_grids=config.model.latents.num_grids,
            conditional_spec=config.model.entropy.conditional_spec,
            mask_config=config.model.entropy.mask_config,
            layers=config.model.entropy.layers,
            context_num_rows_cols=config.model.entropy.context_num_rows_cols,
            shift_log_scale=config.model.entropy.shift_log_scale,
            scale_range=config.model.entropy.scale_range,
            clip_like_cool_chic=config.model.entropy.clip_like_cool_chic,
            use_linear_w_init=config.model.entropy.use_linear_w_init,
            num_knots=config.model.kan.num_knots,
            spline_range=config.model.kan.spline_range,
        )
    else:
        # Use standard MLP-based entropy model
        return functools.partial(
            entropy_models.AutoregressiveEntropyModelConvVideo,
            num_grids=config.model.latents.num_grids,
            conditional_spec=config.model.entropy.conditional_spec,
            mask_config=config.model.entropy.mask_config,
            layers=config.model.entropy.layers,
            activation_fn=config.model.entropy.activation_fn,
            context_num_rows_cols=config.model.entropy.context_num_rows_cols,
            shift_log_scale=config.model.entropy.shift_log_scale,
            scale_range=config.model.entropy.scale_range,
            clip_like_cool_chic=config.model.entropy.clip_like_cool_chic,
            use_linear_w_init=config.model.entropy.use_linear_w_init,
        )
