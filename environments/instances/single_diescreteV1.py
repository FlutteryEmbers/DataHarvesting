from environments.config.single_discrete_v1 import Agent, Task
from environments.config.tasks import Random_Task

task = Task(x_limit=10, y_limit=10, tower_location=[[0, 1], [4, 7], [9, 3]])
task.set_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[30, 30, 30])

Test_Environment = Agent(env=task)

task2 = Random_Task(x_limit=10, y_limit=10)
task2.set_mission(start_at=[0, 0], arrival_at=[9, 9])
DR_Environment = Agent(env = task2)