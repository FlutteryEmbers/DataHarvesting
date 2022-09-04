# from environments.instances.batch_train_v2 import env_list
from environments.instances.loader.test_batch_set3 import env_list
from trainer.DDQN.ddqn import DDQN
from utils import tools, io
from utils import monitor
import sys
from loguru import logger
from datetime import datetime
import numpy as np
import random

random_seeds = [10]
result_saving_iter = 1000
rewards_type = 'Default'
class GameAgent():
    def __init__(self, config, network = 'Default') -> None:
        self.config = config
        self.network = network
        self.timer = tools.Timer()

        now = datetime.now()
        # self.output_dir = 'results/{}'.format(now.strftime("%d-%m-%H-%M-%S"))
        self.output_dir = 'results/'

    def batch_train(self, env_type):
        for i in range(len(env_list.environment_list)):
            env = env_list.get_mission(i)
            env.state_mode = self.network
            output_dir = io.mkdir(self.output_dir + 'batch_train_ddqn/{}/'.format(i))
            self.train_model(env=env, n_games=2000, pre_output_dir=output_dir)
            print('======================================================================================')

    def evaluate_with_model(self, env, model, type_reward, draw_path = False):
        done = False
        s = env.reset()
        episode_reward_sum = 0
        goal = env.goal
        print(goal)
        env.view()
        step = 0
        while not done and (step < env._max_episode_steps or (step < 1000 and draw_path)):
            step += 1
            a = model.choose_action(s, disable_exploration=True)
            s_, r, done, _ = env.step(a, type_reward=type_reward)

            # model.store_transition(s, a, r, s_, done)
            episode_reward_sum += r
            
            s = s_

        return episode_reward_sum, env

    def train_model(self, n_games, env, pre_output_dir, env_type='Default',):
        logger.warning('Training {} Mode'.format(env_type))
        for seed in range(len(random_seeds)):
            tools.setup_seed(random_seeds[seed])
            best_num_steps = float('inf')
            best_rewards = -float('inf')

            output_dir = io.mkdir(pre_output_dir +  '/random_seed_{}/'.format(seed))

            tracker = monitor.Learning_Monitor(output_dir=output_dir, name='ddqn_random_seed_{}'.format(seed), log=['ddqn', env_type], args=self.config)

            logger.warning('Using {} Environment'.format(env.status_tracker.name))
            env.state_mode = self.network
            ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
            # env.mode = 'CNN'
            # ddqn = DDQN_CNN(env=env)
            best_model = None
            for i in range(n_games):
                # logger.success('Start Episode: %s' % i)
                s = env.reset()
                episode_step = 0
                done = False
                self.timer.start()
                while episode_step < env._max_episode_steps and not done:
                    episode_step += 1
                    # env.render()
                    a = ddqn.choose_action(s)
                    s_, r, done, _ = env.step(a, type_reward=rewards_type)

                    ddqn.store_transition(s, a, r, s_, done)
                    # episode_reward_sum += r

                    s = s_

                    # if ddqn.memory_counter > ddqn.memory.mem_size:
                    ddqn.learn()
                    if ddqn.learn_step_counter != 0 and ddqn.learn_step_counter % result_saving_iter == 0:
                        eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward=rewards_type)
                        logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))
                        tracker.store(eval_rewards)

                        if eval_rewards > best_rewards:
                            best_rewards = eval_rewards
                            ddqn.save_models(mode=env_type)

                            _, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward=rewards_type, draw_path=True)
                            logger.warning('best num step: {}'.format(test_env.num_steps))
                            stats = test_env.view()
                            stats.final_reward = eval_rewards
                            stats.save(sub_dir = output_dir, plot = False)

                self.timer.stop()

            x = [i+1 for i in range(n_games)]
            # tools.plot_curve(x, episode_rewards, 'results/' + env_type + '/rewards.png')
            tracker.plot_average_learning_curve(50)
            tracker.plot_learning_curve()
            tracker.dump_to_file()
            tracker.save_log()
            test_env.save_task_info(output_dir)