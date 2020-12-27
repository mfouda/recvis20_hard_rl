import gym
import mpenv.envs
import numpy as np
import matplotlib.pyplot as plt
import time
from A_star import AStarPlanner

env = gym.make('Maze-grid-v3')
bounds = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
start = None

filter_simple = False
num_obstacles = 30 #0.5


#env.reset(start =start, bounds=bounds, filter_simple=filter_simple, num_obstacles=num_obstacles)

iter_ = 0
nb_iters = 1
while iter_ < nb_iters:
    print("==============================")
    print("Number of iterations: ", iter_, "/", nb_iters)

    env.reset(start=start, bounds=bounds, filter_simple=filter_simple, num_obstacles=num_obstacles)

    Edges = env.get_Edges()
    #print("Edges of Obstacles: ", Edges)

    #-----------------------
    #    A star
    #-----------------------

    # start and goal position
    start = env.state.q
    goal = env.goal_state.q

    sx = start[0]  # [m]
    sy = start[1]  # [m]
    gx = goal[0]  # [m]
    gy = goal[1]  # [m]
    grid_size = 3.0  # [m]
    robot_radius = 0.035  # [m]

    sampling_value = 200 # to be changed according to the discretization in nmprepr

    # set obstacle positions
    ox, oy = [], []
    for edge in Edges:
        if edge[0] == edge[2]:
            for i in np.arange(edge[1], edge[3], (edge[3]-edge[1]) / sampling_value):
                ox.append(edge[0])
                oy.append(i)
        elif edge[1] == edge[3]:
            for i in np.arange(edge[0], edge[2], (edge[2] - edge[0]) / sampling_value):
                ox.append(i)
                oy.append(edge[1])
    #print(ox)
    ox_ = [element * sampling_value for element in ox]
    oy_ = [element * sampling_value for element in oy]
    sx_ = sx * sampling_value
    sy_ = sy * sampling_value
    gx_ = gx * sampling_value
    gy_ = gy * sampling_value
    robot_radius_ = robot_radius * sampling_value

    a_star = AStarPlanner(ox_, oy_, grid_size, robot_radius_)
    rx_, ry_ = a_star.planning(sx_, sy_, gx_, gy_)

    if rx_==-999 and ry_==-999:
        continue
    else:
        iter_ += 1

    rx = [element / sampling_value for element in rx_]
    ry = [element / sampling_value for element in ry_]

    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")
    plt.plot(rx, ry, "-r")
    plt.show()

#start_ = 1.4
#end_ = 1.5
#normalize = np.arange(start_,end_,(end_-start_)/len(rx))
error_x = 0.
error_y = 0.

for i in range(len(rx)-2, 0, -1):
    start_state = env.state.q[:2]
    current_state = env.get_state()['current'].q[:2]
    goal_state = env.goal_state.q[:2]

    sign_x = np.sign(rx[i] - rx[i+1])
    sign_y = np.sign(ry[i] - ry[i+1])

    x = rx[i]*sign_x
    y = ry[i]*sign_y
    action = np.array([x, y])

    print("start_state:   ", start_state)
    print("current_state: ", current_state)
    print("goal_state:    ", goal_state)
    print("action:        ", action)

    #env.render()
    env.step(action) # take a action
    time.sleep(0.5)
env.close()


"""
for _ in range(100):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
"""