from environments.v2.game import Agent
# from environments.v1.tasks import Single_Task

class Instances:
    def __init__(self, instance_name, tower_location, start_at, arrival_at, dv_required) -> None:
        self.instance_name = instance_name
        self.phi_model_name = 'configs/config_trans_model_2_D_4.yaml'
        self.x_limit = 10
        self.y_limit = 10
        self.time_scale = 1
        self.max_episode_steps = 100
        print('using {}'.format(instance_name))
        self.environment = Agent(x_limit=self.x_limit, y_limit=self.y_limit, start_at=start_at, arrival_at=arrival_at, \
            tower_location=tower_location, dv_required=dv_required, control_time_scale=self.time_scale, \
            phi_config_file = self.phi_model_name, save_file=instance_name, action_type ='BangSingular')
        

e1 = Instances(instance_name='config_1', tower_location=[[3, 1], [6, 7], [8, 2], [1, 6], [3, 9]],\
               start_at=[[0, 0], [0, 0], [0, 0]], arrival_at=[[9, 6], [5, 5], [7, 8]], dv_required=[5, 6, 3, 4, 4])

# e2 = Instances(instance_name='config_2', tower_location=[[1, 1], [9, 9], [3, 3], [7, 7]],\
#               start_at=[[0, 0], [0, 0], [0, 0]], arrival_at=[[5, 5], [5, 5], [5, 5]], dv_required=[5, 5, 5, 5])

## NOTE: REMOVE CONFIG3
# e3 = Instances(instance_name='config_3', tower_location=[[8, 1], [2, 4], [6, 8]],\
#               start_at=[[0, 0], [0, 0], [0, 0]], arrival_at=[[9, 6], [9, 6], [9, 6]], dv_required=[10, 8, 6])

# e3_d = Instances(instance_name='config_3_less_D', tower_location=[[8, 1], [2, 4], [6, 8]],\
#               start_at=[[0, 0], [0, 0], [0, 0]], arrival_at=[[9, 6], [9, 6], [9, 6]], dv_required=[7, 5, 4])

# e4 = Instances(instance_name='config_4', tower_location=[[8, 1], [2, 4], [9, 5], [3, 8]],\
#                 start_at=[[0, 0], [0, 0], [0, 0]], arrival_at=[[5, 9], [5, 9], [5, 9]], dv_required=[5, 6, 3, 4])

# e5 = Instances(instance_name='config_5', tower_location=[[3, 1], [7, 1], [7, 5], [7, 7]],\
#                 start_at=[[0, 1], [0, 1], [0, 1]], arrival_at=[[7, 9], [7, 9], [7, 9]], dv_required=[5, 6, 3, 3])

# e5x = Instances(instance_name='config_5x', tower_location=[[3, 1], [7, 1], [7, 5], [7, 7]],\
#                 start_at=[[0, 1], [1, 1], [0, 0]], arrival_at=[[7, 9], [7, 8], [6, 9]], dv_required=[5, 6, 3, 3])

# e5s = Instances(instance_name='config_5s', tower_location=[[3, 1], [7, 1], [7, 5], [7, 7]],\
#                 start_at=[[0, 1]], arrival_at=[[7, 9]], dv_required=[4, 3, 3, 3])

env_list = [e1]