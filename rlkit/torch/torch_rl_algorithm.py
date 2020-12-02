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
        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs), save_itrs=True,
        ):
            if self.option is not None and self.option == "cur-v0":
                if epoch % 150 == 0:
                    self.num_obstacles+=1
                    reset_kwargs = {'num_obstacles': self.num_obstacles}
            else:
                reset_kwargs = {}
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
                reset_kwargs=reset_kwargs,
            )
            gt.stamp("evaluation sampling")

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

            self._end_epoch(epoch)


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
