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

"""Config for UVG experiment with KAN instead of MLP."""

from ml_collections import config_dict

from configs import uvg

def get_config() -> config_dict.ConfigDict:
  """Return config object for KAN-based training on UVG dataset."""
  
  # Get original UVG configuration
  config = uvg.get_config()
  exp = config.experiment_kwargs.config
  
  # Add KAN-specific parameters
  exp.model.use_kan = True
  exp.model.kan = config_dict.ConfigDict()
  exp.model.kan.num_knots = 15  # Number of spline knots
  exp.model.kan.spline_range = 3.0  # Range for spline approximation
  
  # Adjust training parameters if needed for KAN
  # exp.training.learning_rate = 5e-4  # Optional adjustment
  
  # Remove the lock to allow modifications
  config.unlock()
  
  return config
