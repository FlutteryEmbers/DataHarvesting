import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../../../..'))

from utils import io, graph
import numpy as np

# position_t = io.load_from_file('assets/V2/DDQN_vs_PPO/P1/Aug18-21_02-ppo_stationary_vanilla/best_case/path.pickle')
position_t = io.load_from_file('assets/V2/DDQN_vs_PPO/P1/ddqn_ma/best_case/path.pickle')
paths = np.transpose(position_t, (1, 0, 2))
graph.plot_path(x_limit = 10, y_limit= 10,\
                                    start_at = [[0, 1], [0, 1], [0, 1]], end_at=[[7, 9], [7, 9], [7, 9]],\
                                    tower_locations=[[3, 1], [7, 1], [7, 5], [7, 7]], agent_paths = paths,\
                                    signal_range = [2, 3, 3.7, 4.3, 4.9], dir='./')