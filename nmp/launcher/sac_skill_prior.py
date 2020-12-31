import gym
import torch
from torch import nn as nn

import sys
sys.path.insert(1, '../../rlkit/')

import rlkit.torch.pytorch_util as ptu
import mpenv.envs
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.samplers.data_collector import (
    GoalConditionedPathCollector,
    MdpPathCollector,
    HRLGoalConditionedPathCollector,
)
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACSkillPriorTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


from nmp.launcher import utils
from spirl.modules.subnetworks import EncoderPointNet, Predictor, BaseProcessingNet
from spirl.utils.pytorch_utils import RemoveSpatial
from spirl.modules.layers import LayerBuilderParams
from nmp.model.recurrent_modules import RecurrentPredictor
from nmp.model.pointnet import PointNet
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import eval_np

from spirl.modules.variational_inference import ProbabilisticModel, Gaussian, MultivariateGaussian, get_fixed_prior, \
                                                mc_kl_divergence
from spirl.utils.general_utils import AttrDict
from spirl.utils.pytorch_utils import map2np, ten2ar, RemoveSpatial, ResizeSpatial, map2torch, find_tensor
import torch.nn as nn
import contextlib
from spirl.utils.pytorch_utils import map2torch, map2np, no_batchnorm_update


class SkillPriorInference(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model.to(device)

    def forward(self, obs):
        z = self.model(obs)
        return MultivariateGaussian(z)

class SkillPriorAgent(nn.Module, ExplorationPolicy):
    def __init__(
        self,
        policy,
        decoder,
        device,
        variant,
    ):
        super().__init__()
        self.device = device
        self.variant = variant


        self.decoder = decoder

        self.decoder_input_initalizer = self._build_decoder_initializer(size=self.variant["decoder_kwargs"]["action_dim"])
        self.decoder_hidden_initalizer = self._build_decoder_initializer(size=self.decoder.cell.get_state_size())

        self.policy = policy.to(device)



    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np, deterministic=deterministic)
        return actions, {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)

    def forward(
        self, obs, reparameterize=True, deterministic=False, return_log_prob=False, act=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        # h = super().forward(obs, return_features=True)
        z = self.policy(obs)
        mean = z.mu

        if not deterministic:
            log_std = z.log_sigma
            log_prob = z.log_prob(mean)
            # sample latent variable
            z_sample = z.sample()
        else:
            log_std = None
            log_prob = None
            z_sample = z.mu
        # decode

        with no_batchnorm_update(self.policy) if obs.shape[0] == 1 else contextlib.suppress():
            actions = self.decode(z_sample,
                                    cond_inputs=None, #self._learned_prior_input(inputs),
                                    steps=self.variant["decoder_kwargs"]["n_rollout_steps"])


        return (actions, z_sample, mean, log_std, log_prob)



    def _build_decoder_initializer(self, size):
        # if self._hp.cond_decode:
        #     # roughly match parameter count of the learned prior
        #     return Predictor(self._hp, input_size=self.prior_input_size, output_size=size,
        #                      num_layers=self._hp.num_prior_net_layers, mid_size=self._hp.nz_mid_prior)
        # else:
        class FixedTrainableInitializer(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device
                self.val = torch.zeros((1, size), requires_grad=True, device=self.device)

            def forward(self, state):
                return self.val.repeat(find_tensor(state).shape[0], 1)
        return FixedTrainableInitializer(self.device)

    def decode(self, z, cond_inputs, steps):
        """Runs forward pass of decoder given skill embedding.
        :arg z: skill embedding
        :arg cond_inputs: info that decoder is conditioned on
        :arg steps: number of steps decoder is rolled out
        """
        ## these to pass also the state to the decoder
        lstm_init_input = self.decoder_input_initalizer(cond_inputs)
        # lstm_init_hidden = self.decoder_hidden_initalizer(cond_inputs)
        return self.decoder(lstm_initial_inputs=AttrDict(x_t=lstm_init_input), #AttrDict(x_t=lstm_init_input),
                            lstm_static_inputs=AttrDict(z=z),
                            steps=steps,
                            ).pred #lstm_hidden_init=lstm_init_hidden


def get_replay_buffer(variant, expl_env):
    """
    Define replay buffer specific to the mode
    """
    mode = variant["mode"]
    if mode == "vanilla":
        replay_buffer = EnvReplayBuffer(
            env=expl_env, **variant["replay_buffer_kwargs"],
        )

    elif mode == "her":
        replay_buffer = ObsDictRelabelingBuffer(
            env=expl_env, **variant["her"], **variant["replay_buffer_kwargs"]
        )

    return replay_buffer


def get_networks(variant, expl_env, device, batch_size):
    """
    Define Q networks and policy network
    """
    qf_kwargs = variant["qf_kwargs"]
    policy_kwargs = variant["policy_kwargs"].copy()
    shared_base = None
    qf_class, qf_kwargs = utils.get_q_network(variant["archi"], qf_kwargs, expl_env)


    # action_dim = env.action_space.low.size
    # obs_dim = env.observation_space.spaces["observation"].low.size
    # goal_dim = env.observation_space.spaces["representation_goal"].low.size
    #
    # kwargs["obs_dim"] = obs_dim + goal_dim
    # kwargs["action_dim"] = action_dim
    #
    #
    #
    # # kwargs["hidden_sizes"] = [kwargs.pop("hidden_dim")] * kwargs.pop("n_layers")
    #
    #
    # robot_props = env.robot_props
    # obs_indices = env.obs_indices
    # obstacles_dim = env.obstacles_dim
    # coordinate_frame = env.coordinate_frame
    #
    #
    #
    #
    # obstacle_point_dim = env.obstacle_point_dim
    # kwargs["q_action_dim"] = 0
    # kwargs["robot_props"] = robot_props
    # kwargs["elem_dim"] = obstacle_point_dim
    # kwargs["input_indices"] = obs_indices
    # kwargs["hidden_activation"] = F.elu
    # kwargs["coordinate_frame"] = coordinate_frame
    # # kwargs["hidden_activation"] = torch.sin
    #
    #

    policy_class, policy_encoder_kwargs = utils.get_policy_network(
        variant["archi"], policy_kwargs, expl_env, "vanilla",
        output_size=variant["policy_kwargs"]["encoder_output_size"],
    ) #"tanhgaussian"


    qf1 = qf_class(**qf_kwargs)
    qf2 = qf_class(**qf_kwargs)
    target_qf1 = qf_class(**qf_kwargs)
    target_qf2 = qf_class(**qf_kwargs)

    # del policy_kwargs["encoder_output_size"]
    policy_encoder = policy_class(**policy_encoder_kwargs)

    ### skill prior
    builder = LayerBuilderParams(use_convs=False, normalization=variant["policy_kwargs"]["normalization"])

    prior = nn.Sequential(
        # ResizeSpatial(self._hp.prior_input_res),
        policy_encoder,
        # RemoveSpatial(),
        BaseProcessingNet(in_dim=variant["policy_kwargs"]["encoder_output_size"],
                          mid_dim=variant["policy_kwargs"]["nz_mid"],
                          out_dim=variant["policy_kwargs"]["nz_vae"] * 2,
                          num_layers=variant["policy_kwargs"]["n_layers_policy"],
                          builder=builder,
                          detached=False,
                          final_activation=None,
                          ),
    )
    prior = SkillPriorInference(prior, device)

    policy = nn.Sequential(
        # ResizeSpatial(self._hp.prior_input_res),
        policy_encoder,
        # RemoveSpatial(),
        BaseProcessingNet(in_dim=variant["policy_kwargs"]["encoder_output_size"],
                          mid_dim=variant["policy_kwargs"]["nz_mid"],
                          out_dim=variant["policy_kwargs"]["nz_vae"] * 2,
                          num_layers=variant["policy_kwargs"]["n_layers_policy"],
                          builder=builder,
                          detached=False,
                          final_activation=None,
                          ),
    )
    policy = SkillPriorInference(policy, device)

    decoder = RecurrentPredictor(nz_mid_lstm=variant["decoder_kwargs"]["nz_mid_lstm"],
                                 n_lstm_layers=variant["decoder_kwargs"]["n_lstm_layers"],
                                 device=device,
                                 batch_size=batch_size,
                                 input_size=variant["decoder_kwargs"]["action_dim"] + variant["policy_kwargs"]["nz_vae"],
                                 output_size=expl_env.action_space.low.size) #variant["decoder_kwargs"]["action_dim"])


    print("Policy:")
    print(policy)

    nets = [qf1, qf2, target_qf1, target_qf2, policy, shared_base, prior, decoder]
    print(f"Q function num parameters: {qf1.num_params()}")
    print(f"policy_encoder num parameters: {policy_encoder.num_params()}")

    return nets


def get_path_collector(variant, expl_env, eval_env, policy, eval_policy):
    """
    Define path collector
    """
    mode = variant["mode"]
    if mode == "vanilla":
        expl_path_collector = MdpPathCollector(expl_env, policy)
        eval_path_collector = MdpPathCollector(eval_env, eval_policy)
    elif mode == "her":
        expl_path_collector = HRLGoalConditionedPathCollector(
            expl_env,
            policy,
            observation_key=variant["her"]["observation_key"],
            desired_goal_key=variant["her"]["desired_goal_key"],
            representation_goal_key=variant["her"]["representation_goal_key"],
        )
        eval_path_collector = HRLGoalConditionedPathCollector(
            eval_env,
            eval_policy,
            observation_key=variant["her"]["observation_key"],
            desired_goal_key=variant["her"]["desired_goal_key"],
            representation_goal_key=variant["her"]["representation_goal_key"],
        )
    return expl_path_collector, eval_path_collector




def sac_skill_prior(variant):
    expl_env = gym.make(variant["env_name"])
    eval_env = gym.make(variant["env_name"])
    expl_env.seed(variant["seed"])
    eval_env.set_eval()

    mode = variant["mode"]
    archi = variant["archi"]
    if mode == "her":
        variant["her"] = dict(
            observation_key="observation",
            desired_goal_key="desired_goal",
            achieved_goal_key="achieved_goal",
            representation_goal_key="representation_goal",
        )

    replay_buffer = get_replay_buffer(variant, expl_env)
    qf1, qf2, target_qf1, target_qf2, policy, shared_base, prior, decoder = get_networks(
        variant, expl_env, device=ptu.device, batch_size=variant["algorithm_kwargs"]["batch_size"],
    )
    skill_prior_policy = SkillPriorAgent(policy=policy, decoder=decoder, device=ptu.device, variant=variant)
    skill_prior = SkillPriorAgent(policy=prior, decoder=decoder, device=ptu.device, variant=variant)

    expl_policy = skill_prior_policy
    eval_policy = MakeDeterministic(skill_prior_policy)

    expl_path_collector, eval_path_collector = get_path_collector(
        variant, expl_env, eval_env, expl_policy, eval_policy
    )

    mode = variant["mode"]
    trainer = SACSkillPriorTrainer(
        env=eval_env,
        policy=skill_prior_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        prior=skill_prior,
        **variant["trainer_kwargs"],
    )
    if mode == "her":
        trainer = HERTrainer(trainer)
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"],
        variant=variant,
    )

    algorithm.to(ptu.device)
    algorithm.train()
