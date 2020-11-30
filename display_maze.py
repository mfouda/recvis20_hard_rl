import gym
import mpenv.envs
import numpy as np


env = gym.make('Maze-grid-v7')
bounds = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
start = np.array([0.5, 0.5, 0, 0, 0, 0, 1])
start = None

filter_simple = False
num_obstacles = 30 #0.5

env.reset(start =start, bounds=bounds, filter_simple=filter_simple, num_obstacles=num_obstacles)
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()