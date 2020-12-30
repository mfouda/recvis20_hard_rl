import abc
from collections import OrderedDict
from typing import Iterable

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch
from torch import nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

import gtimer as gt

import sys
# sys.path.insert(1, '../../nmp/')
# sys.path.insert(1, '../../rlkit/')

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.data_collector import (
    GoalConditionedPathCollector,
    MdpPathCollector,
)
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer

import gym
import numpy as np

def get_path_collector(variant, expl_env, eval_env, policy, eval_policy, grid_size=3):
    """
    Define path collector
    """
    mode = variant["mode"]
    if mode == "vanilla":
        expl_path_collector = MdpPathCollector(expl_env, policy)
        eval_path_collector = MdpPathCollector(eval_env, eval_policy)
    elif mode == "her":
        expl_path_collector = GoalConditionedPathCollector(
            expl_env,
            policy,
            observation_key=variant["her"]["observation_key"],
            desired_goal_key=variant["her"]["desired_goal_key"],
            representation_goal_key=variant["her"]["representation_goal_key"],
            grid_size=grid_size,
        )
        eval_path_collector = GoalConditionedPathCollector(
            eval_env,
            eval_policy,
            observation_key=variant["her"]["observation_key"],
            desired_goal_key=variant["her"]["desired_goal_key"],
            representation_goal_key=variant["her"]["representation_goal_key"],
            grid_size=grid_size,
        )
    return expl_path_collector, eval_path_collector
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


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device, distributed=False):
        for i, net in enumerate(self.trainer.networks):
            net.to(device)
            if distributed:
                self.trainer.networks[i] = DDP(
                    net, device_ids=[device], find_unused_parameters=True
                )

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device, distributed=False):
        networks = self.trainer.networks
        for i, net in enumerate(networks):
            net.to(device.index)
            if distributed:
                networks[i] = DDP(
                    net, device_ids=[device.index], find_unused_parameters=True
                )
        self.trainer.networks = networks

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        self.num_obstacles = 0
        self.upper_x = 0.2
        self.upper_y = 0.2
        grid_size = 1 #start with grid size of 2
        self.bounds = None
        success_rate = 0
        last_update = 0
        self.curr_thresh = 0.7
        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs), save_itrs=True,
        ):
            if self.option is not None and self.option == "cur-v0":
                if epoch % self.cur_range == 0:
                    self.num_obstacles+=1
                    reset_kwargs = {'num_obstacles': self.num_obstacles}
            elif self.option is not None and self.option == "cur-v1": #(epoch - last_update) % (3 * self.cur_range) == 0 or
                if (epoch == 0 or success_rate > self.curr_thresh) and grid_size < self.max_grid_size:
                    grid_size+=1
                    expl_env = gym.make("Maze-grid-v" + str(grid_size))
                    eval_env = gym.make("Maze-grid-v" + str(grid_size))
                    expl_env.seed(self.variant["seed"])
                    eval_env.set_eval()
                    expl_policy = self.policy
                    eval_policy = MakeDeterministic(self.policy)
                    self.replay_buffer = get_replay_buffer(self.variant, expl_env)

                    self.expl_data_collector, self.eval_data_collector = get_path_collector(
                        self.variant, expl_env, eval_env, expl_policy, eval_policy, grid_size
                    )
                    if grid_size < 3:
                        filter_simple = False
                    else:
                        filter_simple = True
                    reset_kwargs = {'filter_simple': filter_simple}
                    last_update = epoch
                    # if self.curr_thresh >= 0.71:
                    #     self.curr_thresh -=0.1
            elif self.option is not None and self.option == "cur-v2":
                if epoch % self.cur_range == 0 or success_rate > 0.8:
                    if self.upper_x <= 0.8:
                        self.upper_x+=0.2
                        self.upper_y+= 0.2
                        self.bounds = np.array(
                            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [self.upper_x, self.upper_y, 0.0, 0.0, 0.0, 0.0, 1.0]]
                        )
                    else:
                        self.bounds = None
                    reset_kwargs = {'bounds': self.bounds}
            elif self.option is not None:
                print("this curr option is not available")
                raise
            else:
                reset_kwargs = {}
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                reset_kwargs=reset_kwargs,
            )
            gt.stamp("evaluation sampling")
            if self.option is not None:
              print("#############")
              print("We are running a custom training with option: ", self.option)
              print("the number of obstacles is: ", self.num_obstacles)
              print("the grid size is: ", grid_size)
              print("curriculum range: ", self.cur_range)
              print("max grid size: ", self.max_grid_size)
              print("bounds :", self.bounds)
              print("#############")
            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                    reset_kwargs=reset_kwargs,
                )
                gt.stamp("exploration sampling", unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp("data storing", unique=False)

                self.training_mode(True)

                for _ in tqdm(range(self.num_trains_per_train_loop), ncols=80):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    gt.stamp("batch sampling", unique=False)
                    self.trainer.train(train_data)
                    gt.stamp("training", unique=False)
                self.training_mode(False)

            stats = self.eval_data_collector.get_diagnostics()
            success_rate = stats["SuccessRate"]
            self._end_epoch(epoch, self.range)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch):
        self._num_train_steps += 1
        torch_batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(torch_batch)

    def eval(self, np_batch):
        self._num_train_steps += 1
        torch_batch = np_to_pytorch_batch(np_batch)
        self.eval_from_torch(torch_batch)

    def get_diagnostics(self):
        return OrderedDict([("num train calls", self._num_train_steps),])

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass

class TorchfDTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch, np_batch_demo, use_bc=False):
        self._num_train_steps += 1
        torch_batch = np_to_pytorch_batch(np_batch)
        torch_batch_demo = np_to_pytorch_batch(np_batch_demo)

        self.train_from_torch(torch_batch, torch_batch_demo, use_bc)

    def eval(self, np_batch):
        self._num_train_steps += 1
        torch_batch = np_to_pytorch_batch(np_batch)
        self.eval_from_torch(torch_batch)

    def get_diagnostics(self):
        return OrderedDict([("num train calls", self._num_train_steps),])

    @abc.abstractmethod
    def train_from_torch(self, batch, batch_demo, use_bc=False):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass

class TorchfDBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device, distributed=False):
        networks = self.trainer.networks
        for i, net in enumerate(networks):
            net.to(device.index)
            if distributed:
                networks[i] = DDP(
                    net, device_ids=[device.index], find_unused_parameters=True
                )
        self.trainer.networks = networks

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        self.num_obstacles = 0
        self.upper_x = 0.2
        self.upper_y = 0.2
        grid_size = 1 #start with grid size of 2
        self.bounds = None
        success_rate = 0
        last_update = 0
        self.curr_thresh = 0.7
        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs), save_itrs=True,
        ):
            if self.option is not None and self.option == "cur-v0":
                if epoch % self.cur_range == 0:
                    self.num_obstacles+=1
                    reset_kwargs = {'num_obstacles': self.num_obstacles}
            elif self.option is not None and self.option == "cur-v1": #(epoch - last_update) % (3 * self.cur_range) == 0 or
                if (epoch == 0 or success_rate > self.curr_thresh) and grid_size < self.max_grid_size:
                    grid_size+=1
                    expl_env = gym.make("Maze-grid-v" + str(grid_size))
                    eval_env = gym.make("Maze-grid-v" + str(grid_size))
                    expl_env.seed(self.variant["seed"])
                    eval_env.set_eval()
                    expl_policy = self.policy
                    eval_policy = MakeDeterministic(self.policy)
                    self.replay_buffer = get_replay_buffer(self.variant, expl_env)

                    self.expl_data_collector, self.eval_data_collector = get_path_collector(
                        self.variant, expl_env, eval_env, expl_policy, eval_policy, grid_size
                    )
                    if grid_size < 3:
                        filter_simple = False
                    else:
                        filter_simple = True
                    reset_kwargs = {'filter_simple': filter_simple}
                    last_update = epoch
                    # if self.curr_thresh >= 0.71:
                    #     self.curr_thresh -=0.1
            elif self.option is not None and self.option == "cur-v2":
                if epoch % self.cur_range == 0 or success_rate > 0.8:
                    if self.upper_x <= 0.8:
                        self.upper_x+=0.2
                        self.upper_y+= 0.2
                        self.bounds = np.array(
                            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [self.upper_x, self.upper_y, 0.0, 0.0, 0.0, 0.0, 1.0]]
                        )
                    else:
                        self.bounds = None
                    reset_kwargs = {'bounds': self.bounds}
            elif self.option is not None:
                print("this curr option is not available")
                raise
            else:
                reset_kwargs = {}
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                reset_kwargs=reset_kwargs,
            )
            gt.stamp("evaluation sampling")
            if self.option is not None:
              print("#############")
              print("We are running a custom training with option: ", self.option)
              print("the number of obstacles is: ", self.num_obstacles)
              print("the grid size is: ", grid_size)
              print("curriculum range: ", self.cur_range)
              print("max grid size: ", self.max_grid_size)
              print("bounds :", self.bounds)
              print("#############")

            if epoch > self.warm_up:
                use_bc = False
            else:
                use_bc = True

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                    reset_kwargs=reset_kwargs,
                )
                gt.stamp("exploration sampling", unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp("data storing", unique=False)

                self.training_mode(True)

                for _ in tqdm(range(self.num_trains_per_train_loop), ncols=80):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    train_data_demo = self.replay_buffer_demo.random_batch(self.batch_size_demo)
                    gt.stamp("batch sampling", unique=False)
                    self.trainer.train(train_data, train_data_demo, use_bc)
                    gt.stamp("training", unique=False)
                self.training_mode(False)

            stats = self.eval_data_collector.get_diagnostics()
            success_rate = stats["SuccessRate"]
            self._end_epoch(epoch, self.range)
