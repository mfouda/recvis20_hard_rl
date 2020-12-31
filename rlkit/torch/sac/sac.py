from collections import OrderedDict

import numpy as np

import rlkit.torch.pytorch_util as ptu
import torch
import torch.optim as optim
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import nn as nn

from rlkit.torch.core import np_to_pytorch_batch




class SACTrainer(TorchTrainer):
    def __init__(
        self,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        render_eval_paths=False,
        alpha=1,
        use_automatic_entropy_tuning=True,
        target_entropy=None,
        policy_optimizer=None,
        qf1_optimizer=None,
        qf2_optimizer=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.alpha = torch.tensor(alpha)
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # heuristic value from Tuomas
                self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)

        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()

        if policy_optimizer is None:
            self.policy_optimizer = optimizer_class(
                self.policy.parameters(), lr=policy_lr,
            )
        if qf1_optimizer is None:
            self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr,)
        if qf2_optimizer is None:
            self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr,)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions), self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = (
            torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            )
            - alpha * new_log_pi
        )

        q_target = (
            self.reward_scale * rewards
            + (1.0 - terminals) * self.discount * target_q_values
        )
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        norm_policy = nn.utils.clip_grad_norm_(self.policy.parameters(), 100)
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        norm_qf1 = nn.utils.clip_grad_norm_(self.qf1.parameters(), 100)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        norm_qf2 = nn.utils.clip_grad_norm_(self.qf2.parameters(), 100)
        self.qf2_optimizer.step()
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics["QF1 Grad Norm"] = ptu.get_numpy(norm_qf1)
            self.eval_statistics["QF2 Grad Norm"] = ptu.get_numpy(norm_qf2)
            self.eval_statistics["Policy Grad Norm"] = ptu.get_numpy(norm_policy)
            self.eval_statistics.update(
                create_stats_ordered_dict("Q1 Predictions", ptu.get_numpy(q1_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q2 Predictions", ptu.get_numpy(q2_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q1 Predictions", ptu.get_numpy(q1_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q2 Predictions", ptu.get_numpy(q2_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q Targets", ptu.get_numpy(q_target),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Log Pis", ptu.get_numpy(log_pi),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Policy mu", ptu.get_numpy(policy_mean),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std", ptu.get_numpy(policy_log_std),
                )
            )
            self.eval_statistics["Alpha"] = alpha.item()
            if self.use_automatic_entropy_tuning:
                self.eval_statistics["Alpha Loss"] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    @networks.setter
    def networks(self, nets):
        self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2 = nets

    def get_snapshot(self):
        snapshot = dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            policy_optimizer=self.policy_optimizer,
            qf1_optimizer=self.qf1_optimizer,
            qf2_optimizer=self.qf2_optimizer,
        )
        for k, v in snapshot.items():
            if "optimizer" not in k:
                if hasattr(v, "module"):
                    snapshot[k] = v.module
        return snapshot


class SACfDTrainer(TorchTrainer):
    def __init__(
        self,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        render_eval_paths=False,
        alpha=1,
        use_automatic_entropy_tuning=True,
        target_entropy=None,
        policy_optimizer=None,
        qf1_optimizer=None,
        qf2_optimizer=None,
        gamma_bc=1,
        bc_dist=False,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.alpha = torch.tensor(alpha)
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # heuristic value from Tuomas
                self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)

        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()

        if policy_optimizer is None:
            self.policy_optimizer = optimizer_class(
                self.policy.parameters(), lr=policy_lr,
            )
        if qf1_optimizer is None:
            self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr,)
        if qf2_optimizer is None:
            self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr,)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.bc_loss = nn.MSELoss()
        self.gamma_bc = gamma_bc
        self.bc_dist = bc_dist
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def train(self, np_batch, batch_demo, use_bc=False):
        self._num_train_steps += 1
        torch_batch = np_to_pytorch_batch(np_batch)
        torch_batch_demo = np_to_pytorch_batch(batch_demo)
        self.train_from_torch(torch_batch, torch_batch_demo, use_bc=use_bc)

    def train_from_torch(self, batch, batch_demo, use_bc=False):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        # demo
        rewards_demo = batch_demo["rewards"]
        terminals_demo = batch_demo["terminals"]
        obs_demo = batch_demo["observations"]
        actions_demo = batch_demo["actions"]
        next_obs_demo = batch_demo["next_observations"]
        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions), self.qf2(obs, new_obs_actions),
        )

        """
        bc loss
        """
        new_obs_actions_demo, policy_mean_demo, policy_log_std_demo, log_pi_demo, *_ = self.policy(
            obs_demo, reparameterize=True, return_log_prob=True,
        )
        if use_bc:
            if self.bc_dist:
                bc_loss = torch.mean(self.bc_loss(policy_mean_demo, actions_demo), dim=-1) + \
                          torch.mean(self.bc_loss(torch.exp(policy_log_std_demo), torch.ones_like(policy_log_std_demo).to(self.device)*0.5), dim=-1)
            else:
                bc_loss = torch.mean(self.bc_loss(new_obs_actions_demo, actions_demo), dim=-1)

            policy_loss = (alpha * log_pi - q_new_actions).mean() + self.gamma_bc * bc_loss
        else:
            bc_loss = torch.tensor([0])
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = (
            torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            )
            - alpha * new_log_pi
        )

        q_target = (
            self.reward_scale * rewards
            + (1.0 - terminals) * self.discount * target_q_values
        )
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        norm_policy = nn.utils.clip_grad_norm_(self.policy.parameters(), 100)
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        norm_qf1 = nn.utils.clip_grad_norm_(self.qf1.parameters(), 100)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        norm_qf2 = nn.utils.clip_grad_norm_(self.qf2.parameters(), 100)
        self.qf2_optimizer.step()
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics["BC Loss"] = np.mean(ptu.get_numpy(bc_loss))
            self.eval_statistics["QF1 Grad Norm"] = ptu.get_numpy(norm_qf1)
            self.eval_statistics["QF2 Grad Norm"] = ptu.get_numpy(norm_qf2)
            self.eval_statistics["Policy Grad Norm"] = ptu.get_numpy(norm_policy)
            self.eval_statistics.update(
                create_stats_ordered_dict("Q1 Predictions", ptu.get_numpy(q1_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q2 Predictions", ptu.get_numpy(q2_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q1 Predictions", ptu.get_numpy(q1_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q2 Predictions", ptu.get_numpy(q2_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q Targets", ptu.get_numpy(q_target),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Log Pis", ptu.get_numpy(log_pi),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Policy mu", ptu.get_numpy(policy_mean),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std", ptu.get_numpy(policy_log_std),
                )
            )
            self.eval_statistics["Alpha"] = alpha.item()
            if self.use_automatic_entropy_tuning:
                self.eval_statistics["Alpha Loss"] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    @networks.setter
    def networks(self, nets):
        self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2 = nets

    def get_snapshot(self):
        snapshot = dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            policy_optimizer=self.policy_optimizer,
            qf1_optimizer=self.qf1_optimizer,
            qf2_optimizer=self.qf2_optimizer,
        )
        for k, v in snapshot.items():
            if "optimizer" not in k:
                if hasattr(v, "module"):
                    snapshot[k] = v.module
        return snapshot




def kl_divergence(q_log_sigma, q_mean, p_log_sigma, p_mean):
    """Here self=q and other=p and we compute KL(q, p)"""
    q_sigma = torch.exp(q_log_sigma)
    p_sigma = torch.exp(p_log_sigma)

    return (p_log_sigma - q_log_sigma) + (q_sigma ** 2 + (q_mean - p_mean) ** 2) \
           / (2 * p_sigma ** 2) - 0.5

class SACSkillPriorTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            soft_target_tau=1e-2,
            target_update_period=1,
            render_eval_paths=False,
            alpha=1,
            use_automatic_entropy_tuning=True,
            target_divergence=None,
            policy_optimizer=None,
            qf1_optimizer=None,
            qf2_optimizer=None,
            prior=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.prior = prior

        self.alpha = torch.tensor(alpha)
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_divergence:
                self.target_divergence = target_divergence
            else:
                # heuristic value from Tuomas
                self.target_divergence = -np.prod(self.env.action_space.shape).item()
            self.alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class([self.alpha], lr=policy_lr)

        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()

        if policy_optimizer is None:
            self.policy_optimizer = optimizer_class(
                self.policy.parameters(), lr=policy_lr,
            )
        if qf1_optimizer is None:
            self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr, )
        if qf2_optimizer is None:
            self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr, )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        """
        Policy and Alpha Loss
        """
        z_policy = self.policy(obs)

        new_obs_actions, policy_mean, policy_log_std, policy_log_pi = z_policy.mu, z_policy.mu, z_policy.log_sigma, \
                                                                          z_policy.log_prob

        _, prior_action, prior_mean, prior_log_std, prior_log_pi, *_  = self.prior(obs)
        kl_div = kl_divergence(policy_log_std, policy_mean, prior_log_std, prior_mean)
        if self.use_automatic_entropy_tuning:
            alpha_loss = (
                    self.alpha * (self.target_divergence - kl_div).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.alpha
        else:
            alpha_loss = 0
            alpha = self.alpha

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions), self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha * kl_div - q_new_actions).mean()
        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        _, new_next_actions, new_policy_mean, _new_policy_log_std, new_policy_log_pi, *_ = self.policy(next_obs)

        _, new_prior_action, new_prior_mean, new_prior_log_std, new_prior_log_pi, *_ = self.prior(next_obs)

        kl_div_next = kl_divergence(_new_policy_log_std, new_policy_mean, new_prior_log_std, new_prior_mean)


        target_q_values = (
                torch.min(
                    self.target_qf1(next_obs, new_next_actions),
                    self.target_qf2(next_obs, new_next_actions),
                )
                - alpha * kl_div_next
        )

        q_target = (
                self.reward_scale * rewards
                + (1.0 - terminals) * self.discount * target_q_values
        )
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        norm_policy = nn.utils.clip_grad_norm_(self.policy.parameters(), 100)
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        norm_qf1 = nn.utils.clip_grad_norm_(self.qf1.parameters(), 100)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        norm_qf2 = nn.utils.clip_grad_norm_(self.qf2.parameters(), 100)
        self.qf2_optimizer.step()
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (alpha * kl_div - q_new_actions).mean()

            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics["QF1 Grad Norm"] = ptu.get_numpy(norm_qf1)
            self.eval_statistics["QF2 Grad Norm"] = ptu.get_numpy(norm_qf2)
            self.eval_statistics["Policy Grad Norm"] = ptu.get_numpy(norm_policy)
            self.eval_statistics.update(
                create_stats_ordered_dict("Q1 Predictions", ptu.get_numpy(q1_pred), )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q2 Predictions", ptu.get_numpy(q2_pred), )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q1 Predictions", ptu.get_numpy(q1_pred), )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q2 Predictions", ptu.get_numpy(q2_pred), )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q Targets", ptu.get_numpy(q_target), )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Log Pis", ptu.get_numpy(new_policy_log_pi), )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Policy mu", ptu.get_numpy(policy_mean), )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std", ptu.get_numpy(policy_log_std),
                )
            )
            self.eval_statistics["Alpha"] = alpha.item()
            if self.use_automatic_entropy_tuning:
                self.eval_statistics["Alpha Loss"] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    @networks.setter
    def networks(self, nets):
        self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2 = nets

    def get_snapshot(self):
        snapshot = dict(
            policy={},
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            policy_optimizer=self.policy_optimizer,
            qf1_optimizer=self.qf1_optimizer,
            qf2_optimizer=self.qf2_optimizer,
        )
        for k, v in snapshot.items():
            if "optimizer" not in k:
                if hasattr(v, "module"):
                    snapshot[k] = v.module
                return snapshot