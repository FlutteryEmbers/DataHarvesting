from environments.instances.batch_train_v3 import env_list
from trainer.DDQN.ddqn import DDQN
from utils import tools, io
from utils import monitor
import sys
from loguru import logger
from datetime import datetime
import numpy as np
import random

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
            output_dir = io.mkdir(self.output_dir + '_' + env_type + '_batch_train_ddqn/{}/'.format(i))
            self.train_model(env=env, n_games=1000, output_dir=output_dir)
            print('======================================================================================')

    def evaluate_with_model(self, env, model, type_reward):
        done = False
        s = env.reset()
        episode_reward_sum = 0
        goal = env.goal
        print(goal)
        env.view()
        step = 0
        while not done and step < 500:
            step += 1
            a = model.choose_action(s, disable_exploration=True)
            s_, r, done, _ = env.step(a, type_reward=type_reward)

            # model.store_transition(s, a, r, s_, done)
            episode_reward_sum += r
            
            s = s_

        return episode_reward_sum, env

    def train_model(self, n_games, env, output_dir, env_type='Default',):
        logger.warning('Training {} Mode'.format(env_type))
        best_num_steps = float('inf')
        best_rewards = -float('inf')

        # output_dir = self.output_dir + '_' + env_type + '_train_ddqn/'

        tracker = monitor.Learning_Monitor(output_dir=output_dir, name='ddqn', log=['ddqn', env_type], args=self.config)

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
                    '''
                    eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn)
                    tracker.store(eval_rewards)
                    logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))
                    test_env.view()
                    '''
                    break

            if i % 50 == 0 or n_games - i < 100:
                eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='Simple')
                logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))

                if eval_rewards > best_rewards:
                    best_rewards = eval_rewards
                    ddqn.save_models(mode=env_type)

                    creward, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='Simple')
                    logger.warning('best num step: {}'.format(test_env.num_steps))
                    print(creward)
                    stats = test_env.view()
                    stats.final_reward = eval_rewards
                    stats.save(sub_dir = output_dir, plot = False)
                '''
                eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='HER')
                tracker.store(eval_rewards)
                logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))
                stats = test_env.view()

                if test_env.num_steps < best_num_steps:
                    logger.warning('best num step: {}'.format(test_env.num_steps))
                    best_num_steps = test_env.num_steps
                    ddqn.save_models(mode=env_type)
                    stats = test_env.view()
                    stats.save(sub_dir = output_dir, plot = False)
                '''
            
            self.timer.stop()

        x = [i+1 for i in range(n_games)]
        # tools.plot_curve(x, episode_rewards, 'results/' + env_type + '/rewards.png')
        tracker.plot_average_learning_curve(50)
        tracker.plot_learning_curve()
        tracker.dump_to_file()
        tracker.save_log()
        test_env.save_task_info(output_dir)