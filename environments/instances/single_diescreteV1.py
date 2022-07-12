from environments.config.single_discrete_v1 import DQN_Environment, Tracker

mission = Tracker(10, 10, [[0, 1], [4, 7], [9, 3]])
mission.set_mission(start_at=[0, 0], arrival_at=[9, 9], dv_required=[30, 50, 80])

Test_Environment = DQN_Environment(mission=mission)