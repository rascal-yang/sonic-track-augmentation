"""Action-level RGB DAgger trainer for a frozen SONIC teacher."""

from copy import deepcopy

import torch

from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer


class TRLDistillTrainer(TRLPPOTrainer):
    """Collect student rollouts and optimize student action means with BC."""

    def _init_trl(self, *args, **kwargs):
        super()._init_trl(*args, **kwargs)
        self.bc_loss_fn = torch.nn.MSELoss()
        self.load_teacher_actor()

    def load_teacher_actor(self):
        path = self.config.network_load_dict.teacher_actor.path
        state = torch.load(path, map_location=self.accelerator.device, weights_only=False)
        state = state.get("policy_state_dict", state.get("actor_model_state_dict", state))
        self.ref_model.load_state_dict(state, strict=True)
        self.ref_model.eval()
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

    def _setup_storage(self):
        super()._setup_storage()
        self.storage.register_key("gt_actions", shape=(self.num_act,), dtype=torch.float)

    def policy_step(self, policy_model, obs_dict, cur_dones=None):
        teacher_actions = self.ref_model.act_inference(deepcopy(obs_dict))
        result = super().policy_step(policy_model, obs_dict, cur_dones=cur_dones)
        result["gt_actions"] = teacher_actions
        return result

    def _get_rollout_data(self, obs_keys):
        data = super()._get_rollout_data(obs_keys)
        data["gt_actions"] = self.storage.gt_actions.transpose(0, 1).contiguous().to(self.accelerator.device)
        return data

    def _get_mb_rollout_data(self, rollout_data, micro_batch_inds):
        base = super()._get_mb_rollout_data(rollout_data, micro_batch_inds)
        base["mb_gt_actions"] = rollout_data["gt_actions"][micro_batch_inds]
        return base

    def _forward_model(self, model, mb_rollout_data):
        result = model.forward(modes=["policy_distill"], input_kwargs={"policy_distill": {
            "obs_dict": mb_rollout_data["mb_obs_dict"],
        }})
        return {"policy_results": result["policy_distill"]}

    def _compute_loss(self, forward_results, mb_rollout_data):
        predicted = forward_results["policy_results"]["action_mean"]
        target = mb_rollout_data["mb_gt_actions"].to(predicted.dtype)
        loss = self.bc_loss_fn(predicted, target)
        return {"loss": loss, "dagger_bc_loss": loss}

    def _update_stats_buffer(self, *args, **kwargs):
        return None

    def _get_train_metrics(self):
        return {"loss/dagger_bc_loss": 0.0}
