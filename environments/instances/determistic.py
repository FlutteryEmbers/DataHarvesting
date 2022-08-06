from environments.config.game import Agent
from environments.config.tasks import Single_Task

task = Single_Task(x_limit=10, y_limit=10, tower_location=[[0, 1], [4, 7], [9, 3]])
task.set_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[30, 30, 30])

Test_Environment = Agent(task=task)

Test_Environment_Continuous = Agent(task=task, action_type='Continuous')
Test_Environment_Eval_Continuous = Agent(task=task, action_type='Continuous')