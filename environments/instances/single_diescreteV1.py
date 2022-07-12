from environments.config.single_discrete_v1 import Agent, Task

task = Task(x_limit=10, y_limit=10, tower_location=[[0, 1], [4, 7], [9, 3]])
task.set_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[30, 50, 80])

Test_Environment = Agent(env=task)