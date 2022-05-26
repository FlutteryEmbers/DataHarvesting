from trainer.ddqn import DDQN, MEMORY_CAPACITY
from environments.single_discrete import DQN_Environment
import signal
from utils.utils import plot_curve

def test_env():
    board = [[3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0]]
    startAt = [0 ,0]
    arrivalAt = [4,4]
    env = DQN_Environment(board=board)
    data_volumn = [200, 500, 800]
    env.init(startAt=startAt, arrivalAt=arrivalAt, data_volume=data_volumn)
    return env

def init_env():
    board = []
    for i in range(10):
        boardrow = []
        for j in range(10):
            boardrow.append(0)
        board.append(boardrow)
    startAt = [0, 0]
    arrivalAt = [9, 9]
    board[0][1] = 1 ## first tower
    board[4][7] = 2 ## second tower
    board[9][3] = 3 ## third tower
    # board[27][27] = 4
    # board[13][12] = 5
    # board[40][30] = 6
    env = DQN_Environment(board=board)
    data_volumn = [300, 500, 800]
    env.init(startAt=startAt, arrivalAt=arrivalAt, data_volume=data_volumn)
    return env

if __name__ == "__main__":                            
    env = init_env()
    # dqn = DQN(5, 5, env.get_action_space().n(), env=env)
    n_games = 1000
    ddqn = DDQN(env.get_linear_state_length(), 5, env)
    best_action_sequence = []
    best_num_steps = 9999999999999999
    episode_rewards = []
    num_steps = []
    for i in range(n_games):
        print('<<<<<<<<<Episode: %s' % i)
        s, current_position = env.reset()
        episode_reward_sum = 0

        while True:
            # env.render()
            a = ddqn.choose_action(s, current_position)
            s_, r, done, current_position = env.step(a)

            '''
            # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            new_r = r1 + r2
            '''

            ddqn.store_transition(s, a, r, s_, done)
            episode_reward_sum += r

            s = s_

            if ddqn.memory_counter > MEMORY_CAPACITY:
                ddqn.learn()

            if done:
                print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                env.view()
                
                break
        num_steps.append(env.num_steps)
        episode_rewards.append(round(episode_reward_sum, 2))
            
    x = [i+1 for i in range(n_games)]
    plot_curve(x, episode_rewards, 'result.png', 1)
    plot_curve(x, num_steps, 'step.png', 2)