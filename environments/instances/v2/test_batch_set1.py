from environments.v2.game import Agent
# from environments.v1.tasks import Single_Task

class Instances:
    def __init__(self) -> None:
        self.instance_name = 'board_loarder_v1'
        self.phi_model_name = 'configs/config_trans_model_2_D_2.yaml'
        self.x_limit = 10
        self.y_limit = 10
        self.tower_location = [[2, 5], [1, 7], [3, 6], [2, 5]]
        self.time_scale = 1
        self.max_episode_steps = 100
        print('using {}'.format(self.instance_name))
        self.environment_list = []

    def add_mission(self, start_at, arrival_at, dv_required):
        print('adding {}'.format(self.instance_name))
        environment = Agent(x_limit=self.x_limit, y_limit=self.y_limit, start_at=start_at, arrival_at=arrival_at, \
            tower_location=self.tower_location, dv_required=dv_required, phi_config_file = self.phi_model_name, save_file=self.instance_name)

        self.environment_list.append(environment)

    def get_mission(self, index):
        return self.environment_list[index]


env_list = Instances()

env_list.add_mission(start_at=[[0, 0]], arrival_at=[[9, 9]], dv_required=[3, 3, 3, 3])

