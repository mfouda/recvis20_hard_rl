from collections import OrderedDict, deque
import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.samplers.rollout_functions import (
    multiagent_multitask_rollout,
    vec_multitask_rollout,
    multitask_rollout,
    rollout,
    hrl_multitask_rollout,
)


class MdpPathCollector(PathCollector):
    def __init__(
        self,
        env,
        policy,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
        self, max_path_length, num_steps, discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length, num_steps - num_steps_collected,
            )
            path = rollout(
                self._env, self._policy, max_path_length=max_path_length_this_loop,
            )
            path_len = len(path["actions"])
            if (
                path_len != max_path_length
                and not path["terminals"][-1]
                and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path["actions"]) for path in self._epoch_paths]
        stats = OrderedDict(
            [
                ("num steps total", self._num_steps_total),
                ("num paths total", self._num_paths_total),
            ]
        )
        stats.update(
            create_stats_ordered_dict(
                "path length", path_lens, always_show_all_stats=True,
            )
        )
        success = [path["rewards"][-1][0] > 0 for path in self._epoch_paths]
        stats["SuccessRate"] = sum(success) / len(success)
        return stats

    def get_snapshot(self):
        return dict(
            # env=self._env,
            policy=self._policy,
        )


class GoalConditionedPathCollector(PathCollector):
    def __init__(
        self,
        env,
        policy,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        observation_key="observation",
        desired_goal_key="desired_goal",
        representation_goal_key="representation_goal",
        grid_size=3,
    ):
        if render_kwargs is None:
            render_kwargs = {}

        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._representation_goal_key = representation_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0
        # self._reset_kwargs = reset_kwargs
        self.grid_size = grid_size

    def collect_new_paths(
        self, max_path_length, num_steps, discard_incomplete_paths, reset_kwargs=None,
    ):
        paths = []
        if reset_kwargs is None:
            reset_kwargs = {}
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length, num_steps - num_steps_collected,
            )
            path = multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                representation_goal_key=self._representation_goal_key,
                return_dict_obs=True,
                reset_kwargs =reset_kwargs
            )
            path_len = len(path["actions"])
            if (
                path_len != max_path_length
                and not path["terminals"][-1]
                and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path["actions"]) for path in self._epoch_paths]
        stats = OrderedDict(
            [
                ("num steps total", self._num_steps_total),
                ("num paths total", self._num_paths_total),
            ]
        )
        stats.update(
            create_stats_ordered_dict(
                "path length", path_lens, always_show_all_stats=True,
            )
        )
        success = [path["env_infos"]["success"][-1] for path in self._epoch_paths]
        stats["SuccessRate"] = sum(success) / len(success)
        stats["grid_size"] = self.grid_size
        return stats

    def get_snapshot(self):
        return dict(
            # env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )


class ParallelGoalConditionedPathCollector(GoalConditionedPathCollector):
    def collect_new_paths(
        self, max_path_length, num_steps, discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        rollouts = None
        obs_reset = None
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length, num_steps - num_steps_collected,
            )
            collected_paths, rollouts, obs_reset = vec_multitask_rollout(
                self._env,
                self._policy,
                rollouts,
                obs_reset,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                representation_goal_key=self._representation_goal_key,
                return_dict_obs=True,
            )
            paths_len = []
            for path in collected_paths:
                path_len = len(path["actions"])
                paths_len.append(path_len)
                num_steps_collected += path_len
                paths.append(path)
            i = np.argmax(paths_len)
            if (
                paths_len[i] != max_path_length
                and not paths[i]["terminals"][-1]
                and discard_incomplete_paths
            ):
                break
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths


class MultiAgentGoalConditionedPathCollector(GoalConditionedPathCollector):
    def __init__(
        self,
        env,
        policy,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        observation_key="observation",
        achieved_q_key="achieved_q",
        desired_q_key="desired_q",
        representation_goal_key="representation_goal",
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._achieved_q_key = achieved_q_key
        self._desired_q_key = desired_q_key
        self._representation_goal_key = representation_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
        self, max_path_length, num_steps, discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length, num_steps - num_steps_collected,
            )
            path_a, path_b = multiagent_multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                achieved_q_key=self._achieved_q_key,
                desired_q_key=self._desired_q_key,
                representation_goal_key=self._representation_goal_key,
            )
            for path in [path_a, path_b]:
                path_len = len(path["actions"])
                if path_len > 0:
                    num_steps_collected += path_len
                    paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_snapshot(self):
        return dict(
            # env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            achieved_q_key=self._achieved_q_key,
            desired_q_key=self._desired_q_key,
        )
class HRLGoalConditionedPathCollector(PathCollector):
    def __init__(
        self,
        env,
        policy,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        observation_key="observation",
        desired_goal_key="desired_goal",
        representation_goal_key="representation_goal",
    ):
        if render_kwargs is None:
            render_kwargs = {}

        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._representation_goal_key = representation_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0
        # self._reset_kwargs = reset_kwargs

    def collect_new_paths(
        self, max_path_length, num_steps, discard_incomplete_paths, reset_kwargs=None,
    ):
        paths = []
        if reset_kwargs is None:
            reset_kwargs = {}
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length, num_steps - num_steps_collected,
            )
            path = hrl_multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                representation_goal_key=self._representation_goal_key,
                return_dict_obs=True,
                reset_kwargs =reset_kwargs
            )
            path_len = len(path["actions"])
            if (
                path_len != max_path_length
                and not path["terminals"][-1]
                and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path["actions"]) for path in self._epoch_paths]
        stats = OrderedDict(
            [
                ("num steps total", self._num_steps_total),
                ("num paths total", self._num_paths_total),
            ]
        )
        stats.update(
            create_stats_ordered_dict(
                "path length", path_lens, always_show_all_stats=True,
            )
        )
        success = [path["env_infos"]["success"][-1] for path in self._epoch_paths]
        stats["SuccessRate"] = sum(success) / len(success)
        return stats

    def get_snapshot(self):
        return dict(
            # env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )