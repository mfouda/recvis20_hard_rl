from collections import OrderedDict

import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
        self, max_replay_buffer_size, observation_dim, action_dim, env_info_sizes,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()
        self._agent_infos = np.zeros((max_replay_buffer_size, 1))

        self._top = 0
        self._size = 0

    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        env_info,
        agent_info,
        **kwargs
    ):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._agent_infos[self._top] = "expert" in agent_info

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            agent_infos=self._agent_infos[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {key: self._env_infos[key][idx] for key in self._env_info_keys}

    def batch_env_info_dict(self, indices):
        return {key: self._env_infos[key][indices] for key in self._env_info_keys}

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        buffer_infos = OrderedDict()
        buffer_infos["Expert_Samples"] = np.sum(self._agent_infos) / self._size
        buffer_infos["Size"] = self._size
        return buffer_infos


class DemoSimpleReplayBuffer(ReplayBuffer):
    def __init__(
        self,demos=None,
    ):
        # self._observation_dim = observation_dim
        # self._action_dim = action_dim
        # self._max_replay_buffer_size = max_replay_buffer_size
        # self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # # It's a bit memory inefficient to save the observations twice,
        # # but it makes the code *much* easier since you no longer have to
        # # worry about termination conditions.
        # self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        # self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # # Make everything a 2D np array to make it easier for other code to
        # # reason about the shape of the data
        # self._rewards = np.zeros((max_replay_buffer_size, 1))
        # # self._terminals[i] = a terminal was received at time i
        # self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        # # Define self._env_infos[key][i] to be the return value of env_info[key]
        # # at time i
        # self._env_infos = {}
        # for key, size in env_info_sizes.items():
        #     self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        # self._env_info_keys = env_info_sizes.keys()
        # self._agent_infos = np.zeros((max_replay_buffer_size, 1))
        #
        # self._top = 0



        obs = []
        for o in  demos["observations"]:
            obs.append(np.concatenate((o["observation"], o["representation_goal"])))
        self._observations = np.array(obs)
        self._actions = np.array(demos["actions"])

        # self._rewards = np.array(demos["rewards"])
        # self._terminals = np.array(demos["Done"])
        #
        # obs = []
        # for o in demos["next_observations"]:
        #     obs.append(np.concatenate((o["observation"], o["representation_goal"])))
        # self._next_obs = np.array(obs)

        self._size = len(self._actions)
        self._agent_infos = np.zeros((self._size, 1))

    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        env_info,
        agent_info,
        **kwargs
    ):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        # self._rewards[self._top] = reward
        # self._terminals[self._top] = terminal
        # self._next_obs[self._top] = next_observation
        #
        # for key in self._env_info_keys:
        #     self._env_infos[key][self._top] = env_info[key]
        self._agent_infos[self._top] = "expert" in agent_info

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            # rewards=self._rewards[indices],
            # terminals=self._terminals[indices],
            # next_observations=self._next_obs[indices],
            agent_infos=self._agent_infos[indices],
        )
        # for key in self._env_info_keys:
        #     assert key not in batch.keys()
        #     batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {key: self._env_infos[key][idx] for key in self._env_info_keys}

    def batch_env_info_dict(self, indices):
        return {key: self._env_infos[key][indices] for key in self._env_info_keys}

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        buffer_infos = OrderedDict()
        buffer_infos["Expert_Samples"] = np.sum(self._agent_infos) / self._size
        buffer_infos["Size"] = self._size
        return buffer_infos
