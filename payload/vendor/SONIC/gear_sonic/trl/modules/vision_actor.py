"""RGB actor used by the SONIC visual-student distillation experiment."""

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal
from torchvision import models

from gear_sonic.trl.modules.base_module import BaseModule
from gear_sonic.utils.running_mean_std import RunningMeanStd


class VisionActor(nn.Module):
    """Maps SONIC proprioception and an RGB frame to the 29-D joint action."""

    is_recurrent = False

    def __init__(self, env_config, algo_config, backbone, module_dim_dict=None,
                 running_mean_std=False, max_rollout_history=1, input_key="actor_obs", **kwargs):
        super().__init__()
        self.input_key = input_key
        self.max_rollout_history = max_rollout_history
        dims = env_config.robot.algo_obs_dim_dict
        module_dim_dict = module_dim_dict or {}
        vision_cfg = backbone.vision_module
        layer_cfg = vision_cfg.module_config_dict.layer_config
        resnet_name = layer_cfg.get("resnet_type", "resnet18")
        resnet = getattr(models, resnet_name)(weights=None)
        feature_dim = 512 if resnet_name in ("resnet18", "resnet34") else 2048
        vision_output_dim = int(module_dim_dict.get("vision_feature_dim", 128))
        self.vision_module = nn.Sequential(
            *list(resnet.children())[:-2],
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(feature_dim, vision_output_dim),
        )
        self.mlp_module = BaseModule(
            obs_dim_dict=dims, module_dim_dict={**module_dim_dict, "vision_feature_dim": vision_output_dim},
            env_config=env_config, algo_config=algo_config,
            process_output_dim=True,
            **backbone.mlp_module,
        )
        self.num_actions = self.mlp_module.output_dim
        self.std = nn.Parameter(float(algo_config.init_noise_std) * torch.ones(self.num_actions))
        if algo_config.get("freeze_noise_std", False):
            self.std.requires_grad = False
        self.distribution = None
        self.obs_dict_buffer = {}
        if running_mean_std:
            self.running_mean_std = RunningMeanStd((dims[input_key],), per_channel=True)
        else:
            self.running_mean_std = None
        Normal.set_default_validate_args(False)

    def forward(self, obs_dict, **kwargs):
        proprio = obs_dict[self.input_key]
        if self.running_mean_std is not None:
            with torch.no_grad():
                proprio = self.running_mean_std(proprio)
        image = obs_dict.get("camera_rgb", obs_dict.get("vision_obs"))
        if image is None:
            raise KeyError("VisionActor requires camera_rgb or vision_obs")
        shape = image.shape
        if image.ndim == 5:
            b, s = shape[:2]
            image = image.reshape(-1, *shape[2:])
            proprio = proprio.reshape(-1, proprio.shape[-1])
        else:
            b, s = image.shape[0], None
        image = image.permute(0, 3, 1, 2).float()
        latent = self.vision_module(image)
        if s is not None:
            latent = latent.reshape(b, s, -1)
        return self.mlp_module.module(torch.cat([proprio, latent], dim=-1))

    def _update_buffer(self, obs_dict):
        for key, value in obs_dict.items():
            self.obs_dict_buffer[key] = value.unsqueeze(1)
        return self.obs_dict_buffer

    def init_rollout(self):
        self.obs_dict_buffer = {}

    def reset(self, dones=None):
        return None

    def clear_rollout(self):
        self.obs_dict_buffer = {}

    def update_distribution(self, obs_dict, last_step_only=False, **kwargs):
        mean = self.forward(obs_dict, **kwargs)
        if last_step_only:
            mean = mean[:, -1]
        self.distribution = Normal(mean, mean * 0.0 + self.std.clamp(min=1e-6))

    def rollout(self, obs_dict, **kwargs):
        self.update_distribution(self._update_buffer(obs_dict), last_step_only=True)
        return {"actions": self.distribution.sample(), "action_mean": self.action_mean,
                "action_sigma": self.action_std}

    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return {"actions": self.distribution.sample(), "action_mean": self.action_mean,
                "action_sigma": self.action_std}

    def act_inference(self, obs_dict, **kwargs):
        self.update_distribution(self._update_buffer(obs_dict), last_step_only=True)
        return self.action_mean

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def has_aux_loss(self):
        return False

    @property
    def has_normalized_actions(self):
        return False

    def to_cpu(self):
        self.vision_module = deepcopy(self.vision_module).cpu()
        self.mlp_module = deepcopy(self.mlp_module).cpu()
        self.std.data = self.std.data.cpu()
