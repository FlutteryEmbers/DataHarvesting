from environments.config.game import Agent
from environments.config.tasks import Single_Task

class Instances:
    def __init__(self) -> None:
        self.x_limit = 10
        self.y_limit = 10
        self.tower_location = [[2, 5], [1, 7], [3, 6], [2, 5]]

        self.task = Single_Task(x_limit=self.x_limit, y_limit=self.y_limit, tower_location=self.tower_location)
        self.task.set_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[3, 3, 3, 3])
        self.prime_environment = Agent(task=self.task, max_episode_steps=300)
        self.environment_list = [self.prime_environment]

    def add_mission(self, start_at, arrival_at, dv_required):
        task = Single_Task(x_limit=self.x_limit, y_limit=self.y_limit, tower_location=self.tower_location)
        task.set_mission(start_at=start_at, arrival_at=arrival_at, dv_required=dv_required)
        environment = Agent(task=task, max_episode_steps=300)
        self.environment_list.append(environment)

    def get_mission(self, index):
        return self.environment_list[index]


env_list = Instances()

env_list.add_mission(start_at=[0, 0], arrival_at=[2, 5], dv_required=[3, 3, 3, 3]) ## Different arrival point
env_list.add_mission(start_at=[0, 0], arrival_at=[3, 2], dv_required=[3, 3, 2, 3]) ## Different arrival point
env_list.add_mission(start_at=[4, 3], arrival_at=[2, 5], dv_required=[3, 2, 2, 3]) ## Different arrival point
env_list.add_mission(start_at=[1, 5], arrival_at=[3, 2], dv_required=[3, 3, 3, 3]) ## Different arrival point


# env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[6, 3, 2.5, 2]) ## Different dv_required
# env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[1, 3.5, 1.5, 3]) ## Different dv_required
# env_list.add_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[4, 3.5, 6, 4]) ## Different dv_required

# env_list.add_mission(start_at=[0, 0], arrival_at=[2, 5], dv_required=[6, 2, 1, 3]) ## Different dv_required & Different Arrival At
# env_list.add_mission(start_at=[3, 5], arrival_at=[6, 3], dv_required=[6, 2, 1, 4])
