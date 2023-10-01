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

class Target_Move_Circular():
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

class Target_Move_Linear():
    def __init__(self, start_at, switch_time, speed) -> None:
        self.start_at = start_at
        self.speed = speed

        self.location = np.array(start_at)
        self.switch_time = np.array(switch_time)
        self.current_speed = np.array(speed)
        self.T = 0

    def update(self, time_scale):
        self.T += time_scale
        if self.T > self.switch_time:
            self.T = 0
            self.current_speed = - self.current_speed
        self.location = self.location + time_scale * self.speed

    def reset(self):
        self.T = 0
        self.current_speed = np.array(self.speed)
        self.location = np.array(self.start_at)

    def set_position(self):
        self.location = self.T * self.speed + self.start_at

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
    
class MA_Discrete():
    def __init__(self, max_speed=1, num_agents = 3):
        # self.time_scale = time_scale
        actions = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]) * max_speed
        self.actions = actions.tolist()
        self.num_agents = num_agents
        self.n = len(self.actions)**num_agents

    def get_action(self, n):
        action_no = []
        while n > 0:
            action_no.append(n%len(self.actions))
            n = n // len(self.actions)
        while len(action_no) < self.num_agents:
            action_no.append(0)
        res = []
        for no in action_no:
            res.append(self.actions[no])

        return res

    def get_actions(self):
        return self.actions

    def sample(self):
        return np.random.randint(0, self.n-1)

class LinearDiscrete():
    def __init__(self, max_speed) -> None:
        self.actions = np.array([[0, 0.1], [0, 0]]) * max_speed
        self.n = len(self.actions)
    
    def get_action(self, n):
        return self.actions[n]

    def sample(self):
        return np.random.randint(0, len(self.actions)-1)


class MA_Continuous():
    def __init__(self, max_speed) -> None:
        self.shape = 2
        self.high = 1
        self.max_speed = max_speed
        self.max_angle = 360

    def get_action(self, action):
        #print('MA')
        #print(action)
        action = action.reshape(-1, self.shape)[:]
        #print(action)
        r = action[:, 0] * self.max_speed
        theta = action[:, 1] * self.max_angle
        # print(r, theta)
        x = r * np.cos(np.radians(theta))
        y = r * np.sin(np.radians(theta))

        coordinates = np.zeros_like(action)
        for i in range(len(coordinates)):
            coordinates[i, 0] = x[i]
            coordinates[i, 1] = y[i]
        # print(x, y)
        # print('--------------')
        
        return coordinates.tolist()

    def sample(self):
        return np.random.rand(2)


class Continuous():
    def __init__(self, max_speed) -> None:
        self.shape = 2
        self.high = 1
        self.max_speed = max_speed
        self.max_angle = 360

    def get_action(self, action):
        # print('SA')
        # print(action)
        r = action[0] * self.max_speed
        theta = action[1] * self.max_angle
        # print(r, theta)
        x = r * math.cos(math.radians(theta))
        y = r * math.sin(math.radians(theta))
        # print(x, y)
        # print('--------------')
        return [[x, y]]

    def sample(self):
        return np.random.rand(2)
    
class BangSingular():
        def __init__(self, max_speed) -> None:
            self.shape = 1
            self.high = 1
            self.max_speed = max_speed
            self.max_angle = 720

        def get_action(self, action):
            thetas = action * self.max_angle
            res = []
            for theta in thetas:
                if theta > 360:
                    res.append([0, 0])
                else:
                    x = self.max_speed * math.cos(math.radians(theta))
                    y = self.max_speed * math.sin(math.radians(theta))
                    res.append([x, y])
            return res
        
class BangSingular2():
        def __init__(self, max_speed) -> None:
            self.shape = 2
            self.high = 1
            self.max_speed = max_speed
            self.max_angle = 360

        def get_action(self, action):
            theta, r = np.split(action, self.shape)
            
            #print(action)
            r = r * self.max_speed
            theta = theta * self.max_angle
            # print(r, theta)
            x = r * np.cos(np.radians(theta))
            y = r * np.sin(np.radians(theta))
            
            return np.dstack((x,y)).squeeze(0).tolist()

Actions = {'Discrete': Discrete, 'Continuous': Continuous, '1D': LinearDiscrete,\
           'MA_Continuous': MA_Continuous, 'BangSingular': BangSingular2, 'MA_Discrete': MA_Discrete}