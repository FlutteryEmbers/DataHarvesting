from environments.config.single_discrete_v1 import Agent
from environments.config.tasks import Random_Task

task2 = Random_Task(x_limit=10, y_limit=10)
task2.set_mission(start_at=[0, 0], arrival_at=[9, 9])
DR_Environment = Agent(env = task2)