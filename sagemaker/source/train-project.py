# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# or in the "license" file accompanying this file. This file is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import os
import json
import gym
import ray

from ray.tune import registry
from ray.rllib.models import ModelCatalog

from procgen_ray_launcher import ProcgenSageMakerRayLauncher

from ray_experiment_builder import RayExperimentBuilder

class MyLauncher(ProcgenSageMakerRayLauncher):
    def register_env_creator(self):
        try:
            from custom.envs.procgen_env_wrapper import ProcgenEnvWrapper
            from custom.envs.cutout_env_wrapper import CutoutEnvWrapper
            from custom.envs.zoom_in_env_wrapper import ZoomInEnvWrapper
        except ModuleNotFoundError:
            from envs.procgen_env_wrapper import ProcgenEnvWrapper
            from envs.cutout_env_wrapper import CutoutEnvWrapper
            from envs.zoom_in_env_wrapper import ZoomInEnvWrapper
        registry.register_env("cutout_env_wrapper", lambda config: CutoutEnvWrapper(config))
        registry.register_env("zoom_in_1.25_env_wrapper", lambda config: ZoomInEnvWrapper(config, 1.25))
        registry.register_env("zoom_in_1.5_env_wrapper", lambda config: ZoomInEnvWrapper(config, 1.5))
        registry.register_env("zoom_in_2.0_env_wrapper", lambda config: ZoomInEnvWrapper(config, 2.0))
        registry.register_env("zoom_in_3.0_env_wrapper", lambda config: ZoomInEnvWrapper(config, 3.0))

    def _get_ray_config(self):
        return {
            "ray_num_cpus": 16,
            "ray_num_gpus": 1,
            "eager": False,
             "v": True, # requried for CW to catch the progress
        }

    def _get_rllib_config(self):
        return {
            "experiment_name": "training",
            "run": "APEX",
            "env": "procgen_env_wrapper",
            "stop": {
                "training_iteration": 400,
            },
            "checkpoint_freq": 50,
            "checkpoint_at_end": True,
            "keep_checkpoints_num": 5,
            "queue_trials": False,
            "config": {
                # === Environment Settings ===
                "gamma": 0.99,
                # === Settings for the Procgen Environment ===
                "env_config": {
                    "env_name": "coinrun",
                    "num_levels": 200,
                    "start_level": 0,
                    "paint_vel_info": False,
                    "use_generated_assets": False,
                    "center_agent": True,
                    "use_sequential_levels": False,
                    "distribution_mode": "easy",
                },
                # === Settings for Resource ===
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "num_cpus_per_worker": 1,
                "num_cpus_for_driver": 1,
                "num_gpus_per_worker": 0,
                "num_gpus": 1,
                # === Settings for Algorithm ===
                "lr": 2.5e-4,
                "adam_epsilon": 1.5e-4,
                "double_q": True,
                "dueling": True,
                "hiddens": [512],
                "prioritized_replay": True,
                "buffer_size": 1000000,
                "worker_side_prioritization": True,
                "n_step": 3,
                "learning_starts": 50000,
                "timesteps_per_iteration": 25000,
                "target_network_update_freq": 50000,
                "min_iter_time_s": 3,
                "rollout_fragment_length": 16,
                "train_batch_size": 512,
                # === Settings for Model ===
                "model": {
                    "custom_model": "my_dqn_nature_cnn",
                    "conv_filters": [[32, [8, 8], 4], [64, [4, 4], 2], [64, [3, 3], 1]],
                    "fcnet_hiddens": [4096]
                    # fcnet_hiddens only indicates the dimension of flattened last conv layer, 
                    # not an actual hidden layer; this is a workaround of current rllib limitation
                    # to achieve the exact dueling network architecture introduced in the original paper.
                },
                # === Exploration Settings ===
                "explore": True,
                "exploration_config": {
                    "type": "PerWorkerEpsilonGreedy"
                },
                # === Evaluation Settings ===
                "evaluation_interval": None,
                "evaluation_num_episodes": 20,
                "evaluation_config": {
                    "env_config": {
                        "env_name": "coinrun",
                        "num_levels": 20,
                        "start_level": 200,
                        "paint_vel_info": False,
                        "use_generated_assets": False,
                        "center_agent": True,
                        "use_sequential_levels": False,
                        "distribution_mode": "easy",
                    },
                    "explore": False
                }
            }
        }
    
    def register_algorithms_and_preprocessors(self):
        try:
            from custom.models.my_dqn_nature_cnn import MyDQNNatureCNN
        except ModuleNotFoundError:
            from models.my_dqn_nature_cnn import MyDQNNatureCNN
        ModelCatalog.register_custom_model("my_dqn_nature_cnn", MyDQNNatureCNN)

    def get_experiment_config(self):
        params = dict(self._get_ray_config())
        params.update(self._get_rllib_config())
        reb = RayExperimentBuilder(**params)
        return reb.get_experiment_definition()


if __name__ == "__main__":
    MyLauncher().train_main()
