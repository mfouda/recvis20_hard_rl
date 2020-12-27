import math
import numpy as np
import matplotlib.pyplot as plt


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                return -999, -999
                #break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        #print(rx,ry)
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

"""
def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 0.84019  # [m]     Start: [0.84019 0.39438]
    sy = 0.39438  # [m]
    gx = 0.33522  # [m]     Goal: [0.33522 0.76823]
    gy = 0.76823  # [m]
    grid_size = 3.0  # [m]
    robot_radius = 0.05  # [m]

    sampling_value = 100

    Edges = [[0. ,     0.     , 0.33333, 0.     ],
            [0.33333, 0.     , 0.66667, 0.     ],
            [0.66667, 0.     , 1.     , 0.     ],
            [0.    ,  0.     , 0.     , 0.33333],
            [0.     , 0.33333, 0.     , 0.66667],
            [0.     , 0.66667, 0.     , 1.     ],
            [0.     , 0.33333, 0.33333, 0.33333],
            [0.     , 1.     , 0.33333, 1.     ],
            [0.66667, 0.     , 0.66667, 0.33333],
            [0.33333, 0.66667, 0.66667, 0.66667],
            [0.66667, 0.33333, 0.66667, 0.66667],
            [0.33333, 1.     , 0.66667, 1.     ],
            [1.     , 0.     , 1.     , 0.33333],
            [1.     , 0.33333, 1.     , 0.66667],
            [0.66667, 1.     , 1.     , 1.     ],
            [1.     , 0.66667, 1.     , 1.     ]]

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

    ox_ = [element * sampling_value for element in ox]
    oy_ = [element * sampling_value for element in oy]
    sx_ = sx * sampling_value
    sy_ = sy * sampling_value
    gx_ = gx * sampling_value
    gy_ = gy * sampling_value
    robot_radius_ = robot_radius * sampling_value

    a_star = AStarPlanner(ox_, oy_, grid_size, robot_radius_)
    rx_, ry_ = a_star.planning(sx_, sy_, gx_, gy_)

    rx = [element / sampling_value for element in rx_]
    ry = [element / sampling_value for element in ry_]

    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")
    plt.plot(rx, ry, "-r")
    plt.show()

if __name__ == '__main__':
    main()
"""