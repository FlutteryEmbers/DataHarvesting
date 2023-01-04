from environments.v2.game import Agent
# from environments.v1.tasks import Single_Task

class Instances:
    def __init__(self) -> None:
        self.instance_name = 'ppo_continuous'
        self.phi_model_name = 'configs/config_trans_model_2_D_4.yaml'
        self.x_limit = 10
        self.y_limit = 10
        self.tower_location=[[3, 1], [6, 7], [8, 2], [1, 6], [3, 9]]
        self.time_scale = 1
        self.max_episode_steps = 100
        print('using {}'.format(self.instance_name))
        self.environment_list = []

    def add_mission(self, start_at, arrival_at, dv_required):
        print('adding {}'.format(self.instance_name))
        args= {'type': 'circular', 'w': [0.1, 0.01, 0.2, 0.03, 0.3], 'w_0': [0, 0, 1, 1, 2], 'radius': [1, 1, 1, 1, 1]}
        environment = Agent(x_limit=self.x_limit, y_limit=self.y_limit, start_at=start_at, arrival_at=arrival_at, \
            tower_location=self.tower_location, dv_required=dv_required, control_time_scale=self.time_scale, \
                phi_config_file = self.phi_model_name, save_file=self.instance_name, action_type ='Continuous', args=args)

        self.environment_list.append(environment)

    def get_mission(self, index):
        return self.environment_list[index]


env_list = Instances()

# env_list.add_mission(start_at=[[0.0, 0.0]], arrival_at=[[9.0, 6.0]], dv_required=[5, 6, 3, 4, 4])
env_list.add_mission(start_at=[[0, 0]], arrival_at=[[9, 6]], dv_required=[5, 6, 3, 4, 4])