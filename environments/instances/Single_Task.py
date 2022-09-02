from environments.config.game import Agent
from environments.config.tasks import Single_Task

Task = Single_Task(x_limit=10, y_limit=10, tower_location=[[0, 5], [4, 7], [9, 3]])
Task.set_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[3, 3, 3])
Test_Environment = Agent(task=Task, max_episode_steps=300)