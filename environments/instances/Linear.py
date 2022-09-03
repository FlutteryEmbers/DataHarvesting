from environments.config.game import Agent
from environments.config.tasks import Single_Task

print('init linear.py')
task = Single_Task(x_limit=1, y_limit=6.01, tower_location=[[0, 2], [0, 2.5], [0, 4.5]])
task.set_mission(start_at=[0, 0], arrival_at=[0, 6], dv_required=[2.3, 2, 1.5])

Test_Environment = Agent(task=task, action_type='1D', max_episode_steps=1000)