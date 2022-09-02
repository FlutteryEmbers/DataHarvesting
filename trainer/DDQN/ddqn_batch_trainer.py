from environments.instances.batch_train_v2 import env_list
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
        self.output_dir = 'results/HER'

    def batch_train(self, env_type):
        for i in range(len(env_list.environment_list)):
            env = env_list.get_mission(i)
            env.state_mode = self.network
            output_dir = io.mkdir(self.output_dir + '_' + env_type + '_batch_train_ddqn/{}/'.format(i))
            self.train_model(env=env, n_games=1000, output_dir=output_dir)
            
    '''
    def evaluate(self, env_type, env = Test_Environment):
        output_dir = self.output_dir + '_' + env_type + '_eval_ddqn/'
        tools.mkdir(output_dir)

        env.state_mode = self.network
        ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
        # env.mode = 'CNN'
        # ddqn = DDQN_CNN(env=env)

        ddqn.load_models(mode=env_type)
        rewards, env = self.evaluate_with_model(env=env, model=ddqn, type_reward='HER')
        print(rewards)
        stats = env.view()
        stats.save(output_dir)
    '''

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
            a = model.choose_action(s, goal, disable_exploration=True)
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
            logger.info('Start Episode: %s' % i)
            # episode_reward_sum = 0
            self.timer.start()
            done = False
            goal = env.goal
            '''
            if i > 1000 and random.random() < 0.30:
                s = random.choice(transitions)[0]
                position = s[:2]
                dv_collected = s[2:]
                s = env.resume(position, dv_collected)
            else:
                s = env.reset()
            '''
            s = env.reset()
            transitions = []
            logger.debug('current goal: {}'.format(goal))
            for p in range(env._max_episode_steps):
                if not done:
                    # env.render()
                    a = ddqn.choose_action(s, goal)
                    s_, r, done, _ = env.step(a, type_reward='HER')

                    ddqn.store_transition(s, a, r, s_, done, goal)
                    transitions.append((s, a, r, s_))
                    # episode_reward_sum += r

                    s = s_

                    # if ddqn.memory_counter > ddqn.memory.mem_size:
                    ddqn.learn()
            
            if not done:
                new_goal = np.copy(s)
                # logger.debug('alternative goal: {}'.format(new_goal))
                if not np.array_equal(new_goal, goal):
                    for p in range(env._max_episode_steps):
                        transition = transitions[p]
                        if np.array_equal(transition[3], new_goal):
                            ddqn.store_transition(transition[0], transition[1], 0.0,
                                                transition[3], True, new_goal)
                            ddqn.learn()
                            break

                        ddqn.store_transition(transition[0], transition[1], transition[2],
                                            transition[3], False, new_goal)
                        ddqn.learn()

            if i % 50 == 0 or n_games - i < 100:
                eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='Simple')
                logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))

                if eval_rewards > best_rewards:
                    best_rewards = eval_rewards
                    ddqn.save_models(mode=env_type)

                    _, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='HER')
                    logger.warning('best num step: {}'.format(test_env.num_steps))
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