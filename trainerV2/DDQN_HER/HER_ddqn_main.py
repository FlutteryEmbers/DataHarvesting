from environments.instances.determistic import Test_Environment, Test_Environment2
from environments.instances.batch_train_v1 import env_list
from trainer.DDQN_HER.HER_ddqn import DDQN
from utils import tools, io
from utils import monitor
import sys
from loguru import logger
from datetime import datetime
import numpy as np

class GameAgent():
    def __init__(self, config, network = 'Default') -> None:
        self.config = config
        self.network = network
        self.timer = tools.Timer()

        now = datetime.now()
        # self.output_dir = 'results/{}'.format(now.strftime("%d-%m-%H-%M-%S"))
        self.output_dir = 'results/HER'

    def evaluate(self, env_type, env = Test_Environment2):
        output_dir = self.output_dir + '_' + env_type + '_eval_ddqn/'
        tools.mkdir(output_dir)

        env.state_mode = self.network
        ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
        # env.mode = 'CNN'
        # ddqn = DDQN_CNN(env=env)

        ddqn.load_models(mode=env_type)
        rewards, env = self.evaluate_with_model(env=env, model=ddqn)
        print(rewards)
        stats = env.view()
        stats.save(output_dir)

    def evaluate_with_model(self, env, model):
        done = False
        s = env.reset()
        episode_reward_sum = 0
        goal = env.goal
        print(goal)
        env.view()
        step = 0
        while not done and step < 1000:
            step += 1
            a = model.choose_action(s, goal, disable_exploration=True)
            s_, r, done, _ = env.step(a, type_reward='HER')

            # model.store_transition(s, a, r, s_, done)
            episode_reward_sum += r
            
            s = s_

        return episode_reward_sum, env

    def train_model(self, n_games, env, env_type):
        logger.warning('Training {} Mode'.format(env_type))
        best_num_steps = float('inf')

        output_dir = self.output_dir + '_' + env_type + '_train_ddqn/'

        tracker = monitor.Learning_Monitor(output_dir=output_dir, name='ddqn', log=['ddqn', env_type], args=self.config)

        logger.warning('Using {} Environment'.format(env.status_tracker.name))
        env.state_mode = self.network
        ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
        # env.mode = 'CNN'
        # ddqn = DDQN_CNN(env=env)
        best_model = None
        for i in range(n_games):
            logger.info('Start Episode: %s' % i)
            s = env.reset()
            # episode_reward_sum = 0

            self.timer.start()
            done = False
            transitions = []
            goal = env.goal
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
                eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn)
                tracker.store(eval_rewards)
                logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))
                test_env.view()

                if test_env.num_steps < best_num_steps:
                    logger.warning('best num step: {}'.format(test_env.num_steps))
                    best_num_steps = test_env.num_steps
                    best_model = ddqn
                    
            
            self.timer.stop()

        x = [i+1 for i in range(n_games)]
        # tools.plot_curve(x, episode_rewards, 'results/' + env_type + '/rewards.png')
        tracker.plot_average_learning_curve(50)
        tracker.plot_learning_curve()
        tracker.dump_to_file()
        tracker.save_log()
        eval_rewards, test_env = self.evaluate_with_model(env=env, model=best_model)
        logger.success('Best Rewards: %s' % (round(eval_rewards, 2)))
        stats = test_env.view()
        stats.save(output_dir)
        best_model.save_models(mode=env_type)
        # tools.plot_curve(x, num_steps, output_dir + 'step.png')

    def train(self, n_games, env_type):
        env = None
        if env_type == 'Default':
            env = Test_Environment2
            logger.warning('max_step_allowed:', env._max_episode_steps)
        else:
            sys.exit("need to set mode")
        self.train_model(n_games, env, env_type)

    def batch_evaluation(self, env_type):
        for i in range(len(env_list.environment_list)):
            env = env_list.get_mission(i)
            env.state_mode = self.network
            output_dir = io.mkdir(self.output_dir + '_' + env_type + '_batch_eval_ddqn/{}/'.format(i))

            ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
            ddqn.load_models(mode=env_type)
            
            rewards, result = self.evaluate_with_model(env=env, model=ddqn)
            print(rewards)
            stats = result.view()
            stats.save(output_dir)
            result.save_task_info(output_dir)

    def train_once(self, env, n_games, output_dir):
        best_num_steps = float('inf')
        ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
        # env.mode = 'CNN'
        # ddqn = DDQN_CNN(env=env)
        best_model = None
        tracker = monitor.Learning_Monitor(output_dir=output_dir, name='ddqn', log=['ddqn'], args=self.config)
        for i in range(n_games):
            logger.info('Start Episode: %s' % i)
            s = env.reset()
            # episode_reward_sum = 0

            self.timer.start()
            done = False
            transitions = []
            goal = env.goal
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
                eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn)
                tracker.store(eval_rewards)
                logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))
                test_env.view()

            # ddqn.lr_decay(n_games)
            # episode_rewards.append(round(episode_reward_sum, 2))
            # num_steps.append(env.num_steps)
            

                if test_env.num_steps < best_num_steps:
                    logger.warning('best num step: {}'.format(test_env.num_steps))
                    best_num_steps = test_env.num_steps
                    best_model = ddqn
                    
            
            self.timer.stop()

        # tools.plot_curve(x, episode_rewards, 'results/' + env_type + '/rewards.png')
        tracker.plot_average_learning_curve(50)
        tracker.plot_learning_curve()
        tracker.dump_to_file()
        tracker.save_log()
        eval_rewards, test_env = self.evaluate_with_model(env=env, model=best_model)
        logger.success('Best Rewards: %s' % (round(eval_rewards, 2)))
        stats = test_env.view()
        stats.save(output_dir)
        test_env.save_task_info(output_dir)

    def batch_train(self, env_type):
        for i in range(len(env_list.environment_list)):
            env = env_list.get_mission(i)
            env.state_mode = self.network
            output_dir = io.mkdir(self.output_dir + '_' + env_type + '_batch_train_ddqn/{}/'.format(i))
            self.train_once(env=env, n_games=1000, output_dir=output_dir)

    def fine_tuning(self, n_game):
        pass
        