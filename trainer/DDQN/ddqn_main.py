from environments.instances.determistic import Test_Environment
from trainer.DDQN.ddqn import DDQN
from utils import tools
from utils import monitor
import sys
from loguru import logger
from datetime import datetime

class DDQN_GameAgent():
    def __init__(self, config, network = 'Default') -> None:
        self.config = config
        self.network = network
        self.timer = tools.Timer()

        now = datetime.now()
        self.output_dir = 'results/{}'.format(now.strftime("%d-%m-%H-%M-%S"))

    def evaluate(self, env_type, env = Test_Environment):
        output_dir = self.output_dir + '_' + env_type + '_eval_ddqn/'
        tools.mkdir(output_dir)

        env.state_mode = self.network
        ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
        # env.mode = 'CNN'
        # ddqn = DDQN_CNN(env=env)

        ddqn.load_models(mode=env_type)
        rewards, env = self.evaluate_with_model(env=env, model=ddqn)

        stats = env.view()
        stats.save(output_dir)

    def evaluate_with_model(self, env, model):
        done = False
        s = env.reset()
        episode_reward_sum = 0

        while not done:
            a = model.choose_action(s, disable_exploration=True)
            s_, r, done, _ = env.step(a, type_reward='HER')
            episode_reward_sum += r
            s = s_

        return episode_reward_sum, env

    def train(self, n_games, env_type):
        logger.warning('Training {} Mode'.format(env_type))
        best_num_steps = float('inf')
        best_rewards = 0
        episode_rewards = []
        num_steps = []

        
        output_dir = self.output_dir + '_' + env_type + '_train_ddqn/'

        tracker = monitor.Learning_Monitor(output_dir=output_dir, name='ddqn', log=['ddqn', env_type], args=self.config)
        env = None
        if env_type == 'Default':
            env = Test_Environment
        elif env_type == 'DR':
            env = DR_Environment
        else:
            sys.exit("need to set mode")

        logger.warning('Using {} Environment'.format(env.status_tracker.name))
        env.state_mode = self.network
        ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
        # env.mode = 'CNN'
        # ddqn = DDQN_CNN(env=env)
        best_model = None
        for i in range(n_games):
            logger.success('Start Episode: %s' % i)
            s = env.reset()
            episode_step = 0

            self.timer.start()
            while episode_step < env._max_episode_steps:
                episode_step += 1
                # env.render()
                a = ddqn.choose_action(s)
                s_, r, done, _ = env.step(a, type_reward='HER')

                ddqn.store_transition(s, a, r, s_, done)
                # episode_reward_sum += r

                s = s_

                # if ddqn.memory_counter > ddqn.memory.mem_size:
                ddqn.learn()
                
                if done:
                    eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn)
                    tracker.store(eval_rewards)
                    logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))
                    test_env.view()
                    break

            # ddqn.lr_decay(n_games)
            # episode_rewards.append(round(episode_reward_sum, 2))
            num_steps.append(env.num_steps)
            
            if  n_games - i < 100:
                if env_type == 'DR':
                    ddqn.save_models(mode=env_type)
                elif env_type == 'Default':
                    if test_env.num_steps < best_num_steps:
                        best_num_steps = test_env.num_steps
                        best_model = ddqn
                        best_model.save_models(mode=env_type)
            
            self.timer.stop()

        x = [i+1 for i in range(n_games)]
        # tools.plot_curve(x, episode_rewards, 'results/' + env_type + '/rewards.png')
        tracker.plot_average_learning_curve(50)
        tracker.plot_learning_curve()
        tracker.dump_to_file()
        tracker.save_log()
        eval_rewards, test_env = self.evaluate_with_model(env=env, model=best_model)
        stats = test_env.view()
        stats.save(output_dir)
        tools.plot_curve(x, num_steps, output_dir + 'step.png')

    def fine_tuning(self, n_game):
        pass
        