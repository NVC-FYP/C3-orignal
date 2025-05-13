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

"""Config for KODAK experiment with KAN instead of MLP."""

from ml_collections import config_dict

from configs import kodak

def get_config() -> config_dict.ConfigDict:
  """Return config object for KAN-based training."""
  
  # Get original Kodak configuration
  config = kodak.get_config()
  exp = config.experiment_kwargs.config
  
  # Add KAN-specific parameters
  exp.model.use_kan_synthesis = True
  exp.model.kan = config_dict.ConfigDict()
  exp.model.kan.num_knots = 10
  exp.model.kan.spline_range = 3.0
  
  # Fix dataset path
  exp.dataset.root_dir = '/home/shittyprogrammers/Desktop/Datasets/Kodak'  # Update to a valid path
  exp.dataset.num_examples = 1 
  
  # Remove the lock to allow modifications
  config.unlock()
  
  return config