from environments.config.game import Agent
from environments.config.tasks import Single_Task

'''
    Advanced Version for Set 3 with modification of data collected and max episode step to ensure that the agent need to wait under the tower
'''
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

    def add_mission(self, tower_location, start_at, arrival_at, dv_required):
        print('adding {}'.format(self.instance_name))
        task = Single_Task(x_limit=self.x_limit, y_limit=self.y_limit, tower_location=tower_location, 
            config_name=self.phi_model_name, save_file_name=self.instance_name + "_{}".format(len(self.environment_list)))
        task.set_mission(start_at=start_at, arrival_at=arrival_at, dv_required=dv_required)
        environment = Agent(task=task, max_episode_steps=self.max_episode_steps, time_scale=self.time_scale)

        self.environment_list.append(environment)

    def get_mission(self, index):
        return self.environment_list[index]


env_list = Instances()

env_list.add_mission(tower_location=[[8, 1], [2, 7]], start_at=[0, 0], arrival_at=[5, 2], dv_required=[5, 5])
env_list.add_mission(tower_location=[[8, 1], [2, 4], [6, 8]], start_at=[0, 0], arrival_at=[9, 6], dv_required=[5, 4, 3])
env_list.add_mission(tower_location=[[8, 1], [2, 4], [9, 5], [3, 8]], start_at=[0, 0], arrival_at=[5, 9], dv_required=[5, 6, 3, 4])
env_list.add_mission(tower_location=[[3, 1], [6, 7], [8, 2], [1, 6], [3, 9]], start_at=[0, 0], arrival_at=[9, 6], dv_required=[5, 6, 3, 4, 4])
env_list.add_mission(tower_location=[[1, 1], [9, 9], [3, 3], [7, 7]], start_at=[0, 0], arrival_at=[5, 5], dv_required=[5, 3, 3, 6])
env_list.add_mission(tower_location=[[3, 1], [7, 1], [7, 5], [7, 7]], start_at=[0, 1], arrival_at=[7, 9], dv_required=[5, 6, 3, 3])
