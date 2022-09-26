class env_info():
    def __init__(self, tower_locations, start_at, arrival_at, signal_range, x_limit = 10, y_limit = 10) -> None:
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.tower_locations = tower_locations
        self.start_at = start_at
        self.arrival_at = arrival_at
        self.signal_range = signal_range

class Instances:
    def __init__(self) -> None:
        self.instance_name = 'board_loader_v6'
        self.phi_model_name = 'configs/config_trans_model_2_D_4.yaml'
        self.x_limit = 10
        self.y_limit = 10
        # self.tower_location = [[7, 2], [1, 8]]
        self.time_scale = 1
        self.max_episode_steps = 100
        print('using {}'.format(self.instance_name))
        self.environment_list = []

    def add_mission(self, tower_location, start_at, arrival_at, signal_range):
        print('adding {}'.format(self.instance_name))
        environment = env_info(tower_locations=tower_location, start_at=start_at, arrival_at=arrival_at, signal_range=signal_range)
        self.environment_list.append(environment)

    def get_mission(self, index):
        return self.environment_list[index]


env_list = Instances()
signal_range = [2, 3, 3.7, 4.3, 4.9]

env_list.add_mission(tower_location=[[8, 1], [2, 7]], start_at=[0, 0], arrival_at=[5, 2], signal_range=signal_range)
env_list.add_mission(tower_location=[[8, 1], [2, 4], [6, 8]], start_at=[0, 0], arrival_at=[9, 6], signal_range=signal_range)
env_list.add_mission(tower_location=[[8, 1], [2, 4], [9, 5], [3, 8]], start_at=[0, 0], arrival_at=[5, 9], signal_range=signal_range)
env_list.add_mission(tower_location=[[3, 1], [6, 7], [8, 2], [1, 6], [3, 9]], start_at=[0, 0], arrival_at=[9, 6], signal_range=signal_range)
env_list.add_mission(tower_location=[[1, 1], [9, 9], [3, 3], [7, 7]], start_at=[0, 0], arrival_at=[5, 5], signal_range=signal_range)
env_list.add_mission(tower_location=[[3, 1], [7, 1], [7, 5], [7, 7]], start_at=[0, 1], arrival_at=[7, 9], signal_range=signal_range)