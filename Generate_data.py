import gym
import mpenv.envs
import numpy as np
import matplotlib.pyplot as plt
import time
from A_star import AStarPlanner

def Generate_Exact_data(env,
                        paths = dict(),
                        output_path="dataset.pkl",
                        bounds = None,
                        start = None,
                        filter_simple = True,
                        num_obstacles = 30,
                        grid_size = 3.0,  # [m]
                        robot_radius = 0.035,  # [m]
                        sampling_value = 175, # to be changed according to the discretization in nmprepr
                        nb_iters=1,
                        verbose=True,  # Print current states and actions
                        plotting=True,
                        time_sleep= False):
    iter_ = 0
    paths["observations"] = []
    paths["actions"] = []
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

        sx = start[0]         # [m]
        sy = start[1]         # [m]
        gx = goal[0]          # [m]
        gy = goal[1]          # [m]

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

        if plotting:
            plt.plot(ox, oy, ".k")
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xb")
            plt.grid(True)
            plt.axis("equal")
            plt.plot(rx, ry, "-r")
            plt.show()

        #-----------------------------------------------------
        #               Generate actions
        #-----------------------------------------------------

        for i in range(len(rx) - 1, 0, -1):

            start_state = env.state.q[:2]
            current_state = env.get_state()['current'].q[:2]
            goal_state = env.goal_state.q[:2]

            obs = env.get_Obs()["observation"]
            observations = {"observation": obs, "representation_goal": goal_state}
            paths["observations"].append(observations)

            x = (rx[i]-current_state[0])/0.18
            y = (ry[i]-current_state[1])/0.18
            action = np.array([x , y ])
            paths["actions"].append(action)

            #action_range = env.robot_props["action_range"]
            #print(action_range)

            if verbose:
                print("start_state:   ", start_state)
                print("current_state: ", current_state)
                print("goal_state:    ", goal_state)
                print("action:        ", action)

            # env.render()
            env.step(action) # take a action
            if time_sleep: time.sleep(0.1)


        #print('********************************************')
        current_state = env.get_state()['current'].q[:2]
        goal_state = env.goal_state.q[:2]
        x = (goal_state[0] - current_state[0]) / 0.18
        y = (goal_state[1] - current_state[1]) / 0.18
        action = np.array([x, y])

        # env.render()
        env.step(action)  # take a action
        if time_sleep: time.sleep(0.1)

        if verbose:
            print("start_state:   ", start_state)
            print("current_state: ", current_state)
            print("goal_state:    ", goal_state)
            print("action:        ", action)

        env.close()
        print("number of transitions: ", len(paths["actions"]))

    # Saving paths
    print("Saving...")
    import pickle
    output_file = open(output_path, 'wb')
    pickle.dump(paths, output_file)
    output_file.close()

env = gym.make('Maze-grid-v5')
Generate_Exact_data(env, nb_iters=2000, plotting=False, verbose=False, time_sleep=False)

#
#import pickle

#with open('dataset.pkl', 'rb') as f:
 #   data = pickle.load(f)
  #  print(len(data["observations"]))
   # print(len(data["actions"]))