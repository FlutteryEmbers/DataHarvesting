import numpy as np
import math

class Obstacles_Avoidance():
    def __init__(self, x_limit, y_limit) -> None:
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.board_info = np.zeros((x_limit, y_limit))
        self.board_info[:, 0] = np.ones(y_limit)
        self.board_info[:, x_limit-1] = np.ones(y_limit)
        self.board_info[0, :] = np.ones(x_limit)
        self.board_info[0, y_limit-1] = np.ones(x_limit)

class Target_Movement_Circular():
    def __init__(self, centers, radius, w, w_0) -> None:
        self.centers = centers
        self.radius = np.array(radius)
        self.w = np.array(w)
        self.T = 0
        self.w_0 = np.array(w_0)
        self.set_position()

    def update(self, time_scale):
        self.T += time_scale
        self.set_position()

    def reset(self):
        self.T = 0
        self.set_position()

    def set_position(self):
        d_y = self.radius*np.cos(self.w*self.T + self.w_0)
        d_x = self.radius*np.sin(self.w*self.T + self.w_0)
        delta = np.transpose([d_x, d_y])
        self.locations = np.array(self.centers) + delta

class Target_Controller():
    def __init__(self, step_sizes, directives, intervals) -> None:
        self.step_sizes = step_sizes
        self.directives = directives
        self.intervals = intervals
        self.time = 0

    def update(self, tower_location):
        location = np.zeros_like(tower_location)
        for i in range(len(tower_location)):
            tower = tower_location[i]
            directive = self.directives[i]
            directive_index = 1
        self.time += 1

    def reset(self):
        self.time = 0

    def lookup(self, direction):
        if direction == 'N':
            return np.array([0, 1])
        elif direction == 'W':
            return np.array([-1, 0])
        elif direction == 'E':
            return np.array([1, 0])
        elif direction == 'S':
            return np.array([0, -1])

class Agent_Controller():
    def __init__(self, time_scale) -> None:
        pass

class Discrete():
    def __init__(self, max_speed=2):
        # self.time_scale = time_scale
        actions = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]) * max_speed
        self.actions = actions.tolist()
        self.n = len(self.actions)

    def get_action(self, n):
        return self.actions[n]

    def get_actions(self):
        return self.actions

    def sample(self):
        return np.random.randint(0, len(self.actions)-1)

class LinearDiscrete():
    def __init__(self, max_speed) -> None:
        self.actions = np.array([[0, 0.1], [0, 0]]) * max_speed
        self.n = len(self.actions)
    
    def get_action(self, n):
        return self.actions[n]

    def sample(self):
        return np.random.randint(0, len(self.actions)-1)


class Continuous():
    def __init__(self, max_speed) -> None:
        self.shape = 2
        self.high = 1
        self.max_speed = max_speed
        self.max_angle = 360

    def get_action(self, action):
        r = action[0] * self.max_speed
        theta = action[1] * self.max_angle
        
        x = r * math.cos(math.radians(theta))
        y = r * math.sin(math.radians(theta))

        return [x, y]

    def sample(self):
        return np.random.rand(2)

Actions = {'Discrete': Discrete, 'Continuous': Continuous, '1D': LinearDiscrete}