import gym
import mpenv.envs
import numpy as np


env = gym.make('Maze-grid-v2')
bounds = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
start = np.array([0.5, 0.5, 0, 0, 0, 0, 1])
start = None
# bounds = None

filter_simple = True
num_obstacles = 2 #0.5

env.reset(start =start, bounds=bounds, filter_simple=filter_simple, num_obstacles=num_obstacles)
print(len(env.action_space.low))
# print(env.action_space.high)
# print(env.action_space.low)

for i in range(400):
    if i % 100 == 0:
        env.reset(start=start, bounds=bounds, filter_simple=filter_simple, num_obstacles=num_obstacles)
    env.render()
    env.step(env.action_space.sample()) # take a random action
    # while True:
    #     print("ok")
env.close()