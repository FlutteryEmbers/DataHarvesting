from environments.v1.game import Agent
from environments.v1.tasks import Single_Task

'''
    Sparse Reward
'''
class Instances:
    def __init__(self) -> None:
        self.instance_name = 'board_loader_v3'
        self.phi_model_name = 'configs/config_trans_model_2_D_3.yaml'
        self.x_limit = 10
        self.y_limit = 10
        self.tower_location = [[7, 2], [1, 8]]
        self.time_scale = 1
        self.max_episode_steps = 50
        print('using {}'.format(self.instance_name))
        self.environment_list = []

    def add_mission(self, start_at, arrival_at, dv_required):
        print('adding {}'.format(self.instance_name))
        task = Single_Task(x_limit=self.x_limit, y_limit=self.y_limit, tower_location=self.tower_location, 
            config_name=self.phi_model_name, save_file_name=self.instance_name)
        task.set_mission(start_at=start_at, arrival_at=arrival_at, dv_required=dv_required)
        environment = Agent(task=task, max_episode_steps=self.max_episode_steps, time_scale=self.time_scale)

        self.environment_list.append(environment)

    def get_mission(self, index):
        return self.environment_list[index]


env_list = Instances()

# env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[1, 1])
env_list.add_mission(start_at=[0, 0], arrival_at=[2, 5], dv_required=[10, 10]) ## Different arrival point
env_list.add_mission(start_at=[0, 0], arrival_at=[3, 2], dv_required=[1, 1]) ## Different arrival point
env_list.add_mission(start_at=[4, 3], arrival_at=[2, 5], dv_required=[1, 1]) ## Different arrival point
env_list.add_mission(start_at=[1, 5], arrival_at=[3, 2], dv_required=[1, 1]) ## Different arrival point

'''
env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[6, 3]) ## Different dv_required
env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[1, 3.5]) ## Different dv_required
env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[4, 3.5]) ## Different dv_required

env_list.add_mission(start_at=[0, 0], arrival_at=[2, 5], dv_required=[6, 2]) ## Different dv_required & Different Arrival At
env_list.add_mission(start_at=[3, 5], arrival_at=[6, 3], dv_required=[6, 2])
'''