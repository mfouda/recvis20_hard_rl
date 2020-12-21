import gym
import mpenv.envs
import numpy as np


env = gym.make('Maze-grid-v3')
bounds = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
start = np.array([0.5, 0.5, 0, 0, 0, 0, 1])
start = None
# bounds = None

filter_simple = True
num_obstacles = 30 #0.5

env.reset(start =start, bounds=bounds, filter_simple=filter_simple, num_obstacles=num_obstacles)
# print(env.action_sace)
# print(env.action_space.high)
# print(env.action_space.low)

for i in range(1000):
    if i % 100 == 0:
        env.reset(start=start, bounds=bounds, filter_simple=filter_simple, num_obstacles=num_obstacles)
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()