import os
import gym
import click
import pickle
from tqdm import tqdm

import torch
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import set_seed, setup_logger
from rlkit.torch.sac.sac import kl_divergence
from rlkit.data_management.simple_replay_buffer import DemoSimpleReplayBuffer
from rlkit.samplers.rollout_functions import (
    multitask_rollout,
    rollout,
)

from nmp import settings
from nmp.policy import utils
from nmp.policy import RandomPolicy, StraightLinePolicy
from nmp.launcher.sac import sac
from nmp.launcher.sac_skill_prior import sac_skill_prior
from nmp.launcher.sac_skill_prior import get_replay_buffer, get_networks, SkillPriorAgent, get_pretrained_networks

from spirl.modules.losses import KLDivLoss, NLL
from spirl.modules.variational_inference import Gaussian

@click.command(help="nmp.train env_name exp_name")
@click.option("-env-name", default='Maze-grid-v3', type=str)
@click.option("-exp-dir", default='maze_baseline', type=str)
@click.option("-s", "--seed", default=None, type=int)
@click.option("-resume", "--resume/--no-resume", is_flag=True, default=False)
@click.option("-mode", "--mode", default="her")
@click.option("-archi", "--archi", default="pointnet")
@click.option("-epochs", "--epochs", default=1, type=int)
@click.option("-rscale", "--reward-scale", default=1, type=float)
@click.option("-h-dim", "--hidden-dim", default=256, type=int)
@click.option("-bs", "--batch-size", default=256, type=int)
@click.option("-lr", "--learning-rate", default=3e-4, type=float)
@click.option("-n-layers", "--n-layers", default=3, type=int)
@click.option("-tau", "--soft-target-tau", default=5e-3, type=float)
@click.option("-auto-alpha", "--auto-alpha/--no-auto-alpha", is_flag=True, default=True)
@click.option("-alpha", "--alpha", default=0.1, type=float)
@click.option("-frac-goal-replay", "--frac-goal-replay", default=0.8, type=float)
@click.option("-horizon", "--horizon", default=80, type=int)
@click.option("-rbs", "--replay-buffer-size", default=int(1e6), type=int)
@click.option("-cpu", "--cpu/--no-cpu", is_flag=True, default=False)
@click.option("-snap-mode","--snapshot-mode",default="last",type=str,help="all, last, gap, gap_and_last, none",)
@click.option("-snap-gap", "--snapshot-gap", default=20, type=int)
@click.option("-option", "--option", default=None, type=str, help='cur-v0 | cur-v1')
@click.option("-cur-range", "--cur-range", default=None, type=int, help='150 | 200 ...')
@click.option("-max-grid-size", "--max-grid-size", default=5, type=int, help='5| 7 ...')

### skill prior
@click.option("-encoder-output-size", "--encoder-output-size", default=64, type=int, help='64')
@click.option("-input-dim", "--input-dim", default=32, type=int, help='5| 7 ...')
@click.option("-num-layers-policy", "--num-layers-policy", default=3, type=int, help='5| 7 ...')
@click.option("-nz_mid", "--nz_mid", default=64, type=int, help='5| 7 ...')
@click.option("-normalization", "--normalization", default="none", type=str, help='none')
@click.option("-nz-vae", "--nz-vae", default=10, type=int, help='10')

## lstm
@click.option("-nz-mid-lstm", "--nz-mid-lstm", default=128, type=int, help='none')
@click.option("-n-lstm-layers", "--n-lstm-layers", default=1, type=int, help='none')
@click.option("-action-dim", "--action-dim", default=0, type=int, help='could be useless')
@click.option("-n-rollout-steps", "--n-rollout-steps", default=10, type=int, help='none')
@click.option("-skill-prior", "--skill-prior", default=False, type=bool, help='none')


@click.option("-range-log", "--range-log", default=1, type=int, help='none')
@click.option("-start-grid-size", "--start-grid-size", default=2, type=int, help='none')

@click.option("-pretrain-path", "--pretrain-path", default=None, type=str, help='none')

@click.option("-no-save-models", "--no-save-models", is_flag=False, default=True)

@click.option("-e", "--episodes", default=1, type=int, help="number of episodes to evaluate")
@click.option("-stoch","--stochastic/--no-stochastic",default=False,is_flag=True,help="stochastic mode",)
@click.option("-exp", "--exp-name", default="", type=str)

@click.option("-demo-path", "--demo-path", default="dataset.pkl", type=str, help='none')


def skill_prior_train(    env_name,
    exp_dir,
    seed,
    resume,
    mode,
    archi,
    epochs,
    reward_scale,
    hidden_dim,
    batch_size,
    learning_rate,
    n_layers,
    soft_target_tau,
    auto_alpha,
    alpha,
    frac_goal_replay,
    horizon,
    replay_buffer_size,
    snapshot_mode,
    snapshot_gap,
    cpu,
    option,
    cur_range,
    max_grid_size,
    encoder_output_size,
    # mlp_output_size,
    nz_mid_lstm,
    n_lstm_layers,
    action_dim,
    nz_vae,
    n_rollout_steps,
    nz_mid,
    input_dim,
    normalization,
    skill_prior,
    num_layers_policy,
    range_log,
    start_grid_size,
    pretrain_path,
    no_save_models,
    demo_path,
    stochastic,
    episodes,
    exp_name,
                          ):

    valid_modes = ["vanilla", "her"]
    valid_archi = [
        "mlp",
        "cnn",
        "pointnet",
    ]
    if mode not in valid_modes:
        raise ValueError(f"Unknown mode: {mode}")
    if archi not in valid_archi:
        raise ValueError(f"Unknown network archi: {archi}")

    machine_log_dir = settings.log_dir()
    exp_dir = os.path.join(machine_log_dir, exp_dir, f"seed{seed}")
    # multi-gpu and batch size scaling
    replay_buffer_size = replay_buffer_size
    num_expl_steps_per_train_loop = 1000
    num_eval_steps_per_epoch = 1000
    min_num_steps_before_training = 1000
    num_trains_per_train_loop = 1000
    # learning rate and soft update linear scaling
    policy_lr = learning_rate
    qf_lr = learning_rate
    if skill_prior:
        action_dim_prior = nz_vae
    else:
        action_dim_prior = None

    variant = dict(
        env_name=env_name,
        algorithm="sac",
        version="normal",
        seed=seed,
        resume=resume,
        mode=mode,
        archi=archi,
        start_grid_size=start_grid_size,
        save_models=no_save_models,
        replay_buffer_kwargs=dict(max_replay_buffer_size=replay_buffer_size,action_dim=action_dim_prior),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            num_epochs=epochs,
            num_eval_steps_per_epoch=num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop=num_expl_steps_per_train_loop,
            num_trains_per_train_loop=num_trains_per_train_loop,
            min_num_steps_before_training=min_num_steps_before_training,
            max_path_length=horizon,
            option=option,
            cur_range=cur_range,
            max_grid_size=max_grid_size,
            range=range_log,    ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=soft_target_tau,
            target_update_period=1,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            use_automatic_entropy_tuning=auto_alpha,
            alpha=alpha,    ),
        qf_kwargs=dict(hidden_dim=hidden_dim, n_layers=n_layers, action_dimension=nz_vae),
        policy_kwargs=dict(hidden_dim=hidden_dim,
                           n_layers=n_layers,
                           # mlp_output_size=mlp_output_size,
                           encoder_output_size=encoder_output_size,
                           input_dim=input_dim,  # dimensionality of the observation input
                           n_layers_policy=num_layers_policy,  # number of policy network layers
                           nz_mid=nz_mid,  # size of the intermediate network layers
                           normalization=normalization,  # normalization used in policy network ['none', 'batch']
                           nz_vae=nz_vae,   ),
        log_dir=exp_dir,
        decoder_kwargs=dict(nz_mid_lstm=nz_mid_lstm,
                            n_lstm_layers=n_lstm_layers,
                            action_dim=action_dim,
                            n_rollout_steps=n_rollout_steps),
        pretrain_path=pretrain_path,
        demo_path=demo_path,    )


    if mode == "her":
        variant["replay_buffer_kwargs"].update(
            dict(
                fraction_goals_rollout_goals=1
                - frac_goal_replay,  # equal to k = 4 in HER paper
                fraction_goals_env_goals=0,
            )
        )
        variant["her"] = dict(
            observation_key="observation",
            desired_goal_key="desired_goal",
            achieved_goal_key="achieved_goal",
            representation_goal_key="representation_goal",
        )

    set_seed(seed)

    """
    setup_logger_kwargs = {
        "exp_prefix": exp_dir,
        "variant": variant,
        "log_dir": exp_dir,
        "snapshot_mode": snapshot_mode,
        "snapshot_gap": snapshot_gap,
    }
    setup_logger(**setup_logger_kwargs)
    """

    #############################################################

    ptu.set_gpu_mode(not cpu, distributed_mode=False)
    print(f"Start training...")

    env = gym.make(variant["env_name"])
    env.seed(variant["seed"])
    env.set_eval()

    _, _, _, _, policy_encoder, _, prior_encoder, decoder = get_networks(
        variant, env, device=ptu.device, batch_size=variant["algorithm_kwargs"]["batch_size"],
    )

    """
    #----------------------------------------------------------------------
    #         Load Pretrained policy to step in the environment
    #----------------------------------------------------------------------
    log_dir = settings.log_dir()

    #exp_name = "/home/alisahili/maze_baseline/seed0_simplev0/params.pkl"
    if exp_name:
        policy_pretrained = utils.load(log_dir, exp_name, cpu, stochastic)
        if stochastic:
            num_params = policy_pretrained.num_params()
        else:
            num_params = policy_pretrained.stochastic_policy.num_params()
        print(f"num params: {num_params}")
    else:
        policy_pretrained = RandomPolicy(env)
    print("Policy: ", policy_pretrained.__dict__)
    """

    #---------------------------------------------------------------------
    #                       training part
    #---------------------------------------------------------------------
    # load data
    demo_file = open(variant["demo_path"], "rb")
    demo_data = pickle.load(demo_file)
    replay_buffer = DemoSimpleReplayBuffer(demo_data)

    alpha_factor = 1
    use_automatic_entropy_tuning = True
    alpha = torch.tensor(alpha_factor)

    if use_automatic_entropy_tuning:
        target_divergence = -np.prod(self.env.action_space.shape).item()
        alpha = ptu.zeros(1, requires_grad=True)
        alpha_optimizer = optim.Adam([alpha], lr=policy_lr)

    qf_criterion = nn.MSELoss()

    policy_optimizer = optim.Adam(policy_encoder.parameters(), lr=policy_lr,)



    for _ in tqdm(range(1), ncols=80): # num_trains_per_train_loop
        train_batch = replay_buffer.random_batch(variant["algorithm_kwargs"]["batch_size"])

        rewards = train_batch["rewards"]
        terminals = train_batch["terminals"]
        obs = train_batch["observations"]
        actions = train_batch["actions"]
        next_obs = train_batch["next_observations"]

        #obs = torch.from_numpy(obs)
        #obs = obs.type(torch.DoubleTensor)
        #obs = obs.cuda()

        #_, new_obs_actions, policy_mean, policy_log_std, policy_log_pi

        z = policy_encoder(obs)
        print(z.shape)
        assert (0)
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

        _, prior_action, prior_mean, prior_log_std, prior_log_pi, _ = prior_encoder(obs)

        recons_out = decoder(   lstm_initial_inputs=None,  # AttrDict(x_t=lstm_init_input),
                                lstm_static_inputs=AttrDict(z=z_sample),
                                teps=self.variant["decoder_kwargs"]["n_rollout_steps"],
                            ).pred

        # reconstruction loss, assume unit variance model output Gaussian
        reconstruction_mse_weight = 1.
        recons_loss = NLL(reconstruction_mse_weight)(Gaussian(recons_out, torch.zeros_like(recons_out)), actions)

        # KL loss
        kl_div_weight = 1.
        kl_loss = KLDivLoss(kl_div_weight)(model_output.q, model_output.p)

        #
        nll_prior_train = True
        if nll_prior_train:
            loss = NLL(breakdown=0)(model_output.q_hat, model_output.z_q.detach())
        else:
            loss = KLDivLoss(breakdown=0)(model_output.q.detach(), model_output.q_hat)
        # aggregate loss breakdown for each of the priors in the ensemble
        #n_prior_nets = 1.
        breakdown_loss = loss.mean() #torch.stack([chunk.mean() for chunk in torch.chunk(breakdown_loss, n_prior_nets)])


skill_prior_train()