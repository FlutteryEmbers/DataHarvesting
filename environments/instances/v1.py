from environments.config.game import Agent
from environments.config.tasks import Single_Task

class Instances:
    def __init__(self) -> None:
        self.task = Single_Task(x_limit=10, y_limit=10, tower_location=[[0, 1], [4, 7], [9, 3]])
        self.task.set_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[30, 30, 30])
        self.prime_environment = Agent(task=self.task)
        self.environment_list = [self.prime_environment]

    def add_mission(self, start_at, arrival_at, dv_required):
        task = Single_Task(x_limit=10, y_limit=10, tower_location=[[0, 1], [4, 7], [9, 3]])
        task.set_mission(start_at=start_at, arrival_at=arrival_at, dv_required=dv_required)
        environment = Agent(task=task)
        self.environment_list.append(environment)

    def get_mission(self, index):
        return self.environment_list[index]


env_list = Instances()
env_list.add_mission(start_at=[0, 0], arrival_at=[2, 5], dv_required=[30, 30, 30]) ## Different arrival point
env_list.add_mission(start_at=[0, 0], arrival_at=[3, 2], dv_required=[30, 30, 30]) ## Different arrival point

env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[30, 30, 25]) ## Different dv_required
env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[10, 35, 15]) ## Different dv_required
env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[40, 35, 40]) ## Different dv_required

env_list.add_mission(start_at=[0, 0], arrival_at=[2, 5], dv_required=[20, 20, 20]) ## Different dv_required & Different Arrival At
