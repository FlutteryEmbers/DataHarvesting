from environments.instances.determistic import Test_Environment
from environments.instances.randomized import DR_Environment
from trainer.Q_Learning.ddqn import DDQN
# from trainer.Q_Learning.ddqn_cnn import DDQN_CNN
from utils import tools
import sys

class DDQN_GameAgent():
    def __init__(self, config, network = 'Default') -> None:
        self.config = config
        self.network = network
        self.timer = tools.Timer()

    def evaluate(self, env_type, env = Test_Environment):
        env.state_mode = self.network
        ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
        # env.mode = 'CNN'
        # ddqn = DDQN_CNN(env=env)

        ddqn.load_models(mode=env_type)
        done = False
        s = env.reset()
        episode_reward_sum = 0

        while not done:
            a = ddqn.choose_action(s, disable_exploration=True)
            s_, r, done, _ = env.step(a)

            ddqn.store_transition(s, a, r, s_, done)
            episode_reward_sum += r

            s = s_

        stats = env.view()
        stats.save(env_type + "/")

    def train(self, n_games, env_type):
        print('training in mode: ' + env_type)
        best_num_steps = float('inf')
        best_rewards = 0
        episode_rewards = []
        num_steps = []

        env = None
        if env_type == 'Default':
            env = Test_Environment
        elif env_type == 'DR':
            env = DR_Environment
        else:
            sys.exit("need to set mode")

        env.state_mode = self.network
        ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
        # env.mode = 'CNN'
        # ddqn = DDQN_CNN(env=env)

        for i in range(n_games):
            print('<<<<<<<<<Episode: %s' % i)
            s = env.reset()
            episode_reward_sum = 0

            self.timer.start()
            while True:
                # env.render()
                a = ddqn.choose_action(s)
                s_, r, done, _ = env.step(a)

                ddqn.store_transition(s, a, r, s_, done)
                episode_reward_sum += r

                s = s_

                if ddqn.memory_counter > ddqn.memory.mem_size:
                    ddqn.learn()

                if done:
                    print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                    env.view()
                    break

            episode_rewards.append(round(episode_reward_sum, 2))
            num_steps.append(env.num_steps)
            
            if  n_games - i < 100:
                if env_type == 'DR':
                    ddqn.save_models(mode=env_type)
                elif env_type == 'Default':
                    if env.num_steps < best_num_steps:
                        best_num_steps = env.num_steps
                        ddqn.save_models(mode=env_type)
            
            self.timer.stop()

        x = [i+1 for i in range(n_games)]
        tools.plot_curve(x, episode_rewards, 'results/' + env_type + '/rewards.png')
        tools.plot_curve(x, num_steps, 'results/' + env_type + '/step.png')

    def fine_tuning(self, n_game):
        pass
        