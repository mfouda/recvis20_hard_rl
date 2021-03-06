import abc

from tqdm import tqdm

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: PathCollector,
        evaluation_data_collector: PathCollector,
        replay_buffer: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
        option=None,
        variant=None,
        cur_range=1500,
        max_grid_size=7,
        range=1,
        replay_buffer_demo=None,
        warm_up=100,
        batch_size_demo=64,
        save_models=True,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.option = option
        self.variant = variant
        self.cur_range = cur_range
        self.max_grid_size = max_grid_size
        self.range = range
        self.replay_buffer_demo = replay_buffer_demo
        self.warm_up = warm_up
        self.batch_size_demo = batch_size_demo
        self.save_models = save_models

    def _train(self):
        """ should be implemented in the inherited class"""
        pass
