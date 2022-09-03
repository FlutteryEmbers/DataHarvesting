from environments.config.game import Agent
from environments.config.tasks import Single_Task

class Instances:
    def __init__(self) -> None:
        print('using batch_train_v2.py')
        self.task = Single_Task(x_limit=10, y_limit=10, tower_location=[[0, 5], [4, 7], [9, 3]])
        self.task.set_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[3, 3, 3])
        self.prime_environment = Agent(task=self.task, max_episode_steps=300)
        self.environment_list = [self.prime_environment]

    def add_mission(self, start_at, arrival_at, dv_required):
        print('load batch_train_v2.py')
        task = Single_Task(x_limit=10, y_limit=10, tower_location=[[0, 5], [4, 7], [9, 3]])
        task.set_mission(start_at=start_at, arrival_at=arrival_at, dv_required=dv_required)
        environment = Agent(task=task, max_episode_steps=300)
        self.environment_list.append(environment)

    def get_mission(self, index):
        return self.environment_list[index]


env_list = Instances()
env_list.add_mission(start_at=[0, 0], arrival_at=[2, 5], dv_required=[3, 3, 3]) ## Different arrival point
env_list.add_mission(start_at=[0, 0], arrival_at=[3, 2], dv_required=[3, 3, 3]) ## Different arrival point

env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[3, 3, 2.5]) ## Different dv_required
env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[1, 3.5, 1.5]) ## Different dv_required
env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[4, 3.5, 4.0]) ## Different dv_required

env_list.add_mission(start_at=[0, 0], arrival_at=[2, 5], dv_required=[6, 2, 1]) ## Different dv_required & Different Arrival At