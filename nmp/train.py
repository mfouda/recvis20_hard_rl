import os
import click

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import set_seed, setup_logger

from nmp.launcher.sac import sac
from nmp.launcher.sac_skill_prior import sac_skill_prior
from nmp import settings
from nmp.launcher.sac_rnd import sac_rnd

@click.command(help="nmp.train env_name exp_name")
@click.option("-env-name", default='Maze-grid-v3', type=str)
@click.option("-exp-dir", default='maze_baseline', type=str)
@click.option("-s", "--seed", default=None, type=int)
@click.option("-resume", "--resume/--no-resume", is_flag=True, default=False)
@click.option("-mode", "--mode", default="her")
@click.option("-archi", "--archi", default="pointnet")
@click.option("-epochs", "--epochs", default=3000, type=int)
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
@click.option(
    "-snap-mode",
    "--snapshot-mode",
    default="last",
    type=str,
    help="all, last, gap, gap_and_last, none",
)
@click.option("-snap-gap", "--snapshot-gap", default=20, type=int)
@click.option("-option", "--option", default=None, type=str, help='cur-v0 | cur-v1')
@click.option("-cur-range", "--cur-range", default=None, type=int, help='150 | 200 ...')
@click.option("-max-grid-size", "--max-grid-size", default=None, type=int, help='5| 7 ...')
### skill prior
@click.option("-encoder-output-size", "--encoder-output-size", default=None, type=int, help='64')
@click.option("-input-dim", "--input-dim", default=32, type=int, help='5| 7 ...')
@click.option("-num-layers-policy", "--num-layers-policy", default=3, type=int, help='5| 7 ...')
@click.option("-nz_mid", "--nz_mid", default=64, type=int, help='5| 7 ...')
@click.option("-normalization", "--normalization", default="none", type=str, help='none')
@click.option("-nz-vae", "--nz-vae", default=None, type=int, help='10')
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

@click.option("-deep-pointnet", "--deep-pointnet", is_flag=True, default=False)
## noisy nets
@click.option("-noisy", "--noisy", is_flag=True, default=False)
@click.option("-sigma-init", "--sigma-init", default=0.017, type=float, help='noisy sigma')

## explore
@click.option("-rnd-explore", "--rnd-explore", is_flag=True, default=False)



def main(
    env_name,
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
    deep_pointnet,
    noisy,
    sigma_init,
    rnd_explore,
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
            range=range_log,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=soft_target_tau,
            target_update_period=1,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            use_automatic_entropy_tuning=auto_alpha,
            alpha=alpha,
            noisy=noisy,
        ),
        qf_kwargs=dict(hidden_dim=hidden_dim, n_layers=n_layers, action_dimension=nz_vae, sigma_init=sigma_init),
        policy_kwargs=dict(hidden_dim=hidden_dim,
                           n_layers=n_layers,
                           # mlp_output_size=mlp_output_size,
                           encoder_output_size=encoder_output_size,
                           input_dim=input_dim,  # dimensionality of the observation input
                           n_layers_policy=num_layers_policy,  # number of policy network layers
                           nz_mid=nz_mid,  # size of the intermediate network layers
                           normalization=normalization,  # normalization used in policy network ['none', 'batch']
                           nz_vae=nz_vae,
                           deep_pointnet=deep_pointnet,
                           sigma_init=sigma_init,
     ),
        log_dir=exp_dir,
        decoder_kwargs=dict(nz_mid_lstm=nz_mid_lstm,
                            n_lstm_layers=n_lstm_layers,
                            action_dim=action_dim,
                            n_rollout_steps=n_rollout_steps),
        pretrain_path=pretrain_path,

    )
    if mode == "her":
        variant["replay_buffer_kwargs"].update(
            dict(
                fraction_goals_rollout_goals=1
                - frac_goal_replay,  # equal to k = 4 in HER paper
                fraction_goals_env_goals=0,
            )
        )
    set_seed(seed)

    setup_logger_kwargs = {
        "exp_prefix": exp_dir,
        "variant": variant,
        "log_dir": exp_dir,
        "snapshot_mode": snapshot_mode,
        "snapshot_gap": snapshot_gap,
    }
    setup_logger(**setup_logger_kwargs)
    ptu.set_gpu_mode(not cpu, distributed_mode=False)
    print(f"Start training...")
    if skill_prior:
        print("skill prior training ...")
        sac_skill_prior(variant)
    elif rnd_explore:
        sac_rnd(variant)
    else:
        sac(variant)


if __name__ == "__main__":
    main()
