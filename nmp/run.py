import click
import numpy as np
import gym

import mpenv.envs

from nmp.policy import utils
from nmp.policy import RandomPolicy, StraightLinePolicy
from nmp import settings

from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.launchers.launcher_util import set_seed
from rlkit.samplers.rollout_functions import (
    multitask_rollout,
    rollout,
)


@click.command()
@click.argument("env_name", type=str)
@click.option("-exp", "--exp-name", default="", type=str)
@click.option("-s", "--seed", default=None, type=int)
@click.option("-h", "--horizon", default=50, type=int, help="max steps allowed")
@click.option("-n", "--nb-paths", default=100, type=int, help="number of paths to save")
@click.option(
    "-e", "--episodes", default=0, type=int, help="number of episodes to evaluate"
)
@click.option("-cpu", "--cpu/--no-cpu", default=False, is_flag=True, help="use cpu")
@click.option(
    "-stoch",
    "--stochastic/--no-stochastic",
    default=False,
    is_flag=True,
    help="stochastic mode",
)

@click.option("-n", "--output-path", default='nmp/data/data.pkl',
              type=str, help="output path")
@click.option("-perfect", "--perfect", default=False, is_flag=True, help="output path")
@click.option("-render-gen-data", "--render-gen-data", default=False, is_flag=True, help="render_gen_data")

def main(env_name, exp_name, seed, horizon, nb_paths, episodes, cpu, stochastic,
         output_path, perfect, render_gen_data):
    if not cpu:
        set_gpu_mode(True)
    set_seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    env.set_eval()
    log_dir = settings.log_dir()

    if exp_name:
        policy = utils.load(log_dir, exp_name, cpu, stochastic)
        if stochastic:
            num_params = policy.num_params()
        else:
            num_params = policy.stochastic_policy.num_params()
        print(f"num params: {num_params}")
    else:
        policy = RandomPolicy(env)

    render = episodes == 0

    reset_kwargs = {}

    def rollout_fn():
        return multitask_rollout(
            env,
            policy,
            horizon,
            render=render_gen_data,
            observation_key="observation",
            desired_goal_key="desired_goal",
            representation_goal_key="representation_goal",
            **reset_kwargs,
        )

    print("number of paths: ", nb_paths)
    if render:
        paths = utils.render(env, rollout_fn, nb_paths, output_path=output_path,
                             perfect=perfect, render_gen_data=render_gen_data)
    else:
        success_rate, n_col, paths_states = utils.evaluate(rollout_fn, episodes)
        print(f"Success rate: {success_rate} - Collisions: {n_col}")


if __name__ == "__main__":
    main()
