#from environments.instances.batch_train_v3 import env_list
from environments.instances.loader.test_batch_set5 import env_list
from trainer.DDQN_HER.HER_ddqn import DDQN
from utils import tools, io
from utils import monitor
import sys
from loguru import logger
from datetime import datetime
import numpy as np
import random

# random_seed = [10, 20, 30, 40, 50, 66, 88, 120, 240, 360, 245, 670, 890]
random_seed = [20]
result_saving_iter = 1000
n_game = 3000

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
            output_dir = io.mkdir(self.output_dir +  env_list.instance_name + '/batch_train_ddqn_her/{}/'.format(i))
            self.train_model(env=env, n_games=n_game, pre_output_dir=output_dir)
            
    def evaluate_with_model(self, env, model, type_reward):
        done = False
        s = env.reset()
        episode_reward_sum = 0
        goal = env.goal
        print(goal)
        env.view()
        step = 0
        while not done and step < env._max_episode_steps:
            step += 1
            a = model.choose_action(s, goal, disable_exploration=True)
            s_, r, done, _ = env.step(a, type_reward=type_reward)

            # model.store_transition(s, a, r, s_, done)
            episode_reward_sum += r
            
            s = s_

        return episode_reward_sum, env

    def train_model(self, n_games, env, pre_output_dir, env_type='Default',):
        logger.warning('Training {} Mode'.format(env_type))
        for seed in range(len(random_seed)):
            tools.setup_seed(random_seed[seed])
            best_num_steps = float('inf')
            best_rewards = -float('inf')
            output_dir = io.mkdir(pre_output_dir +  '/random_seed_{}/'.format(seed))

            tracker = monitor.Learning_Monitor(output_dir=output_dir, name='ddqn_random_seed_{}'.format(seed), log=['ddqn', env_type], args=self.config)

            logger.warning('Using {} Environment'.format(env.status_tracker.name))
            env.state_mode = self.network
            ddqn = DDQN(env=env, config = self.config['AGENT'], network_config=self.config['NETWORK'])
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
                        if ddqn.learn_step_counter != 0 and ddqn.learn_step_counter % result_saving_iter == 0:
                            eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='HER')
                            logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))
                            tracker.store(eval_rewards)

                            if eval_rewards > best_rewards:
                                best_rewards = eval_rewards
                                ddqn.save_models(mode=env_type)

                                _, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='HER')
                                logger.warning('best num step: {}'.format(test_env.num_steps))
                                stats = test_env.view()
                                stats.final_reward = eval_rewards
                                stats.save(sub_dir = output_dir, plot = False)
                
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
                                if ddqn.learn_step_counter != 0 and ddqn.learn_step_counter % result_saving_iter == 0:
                                    eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='HER')
                                    logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))
                                    tracker.store(eval_rewards)

                                    if eval_rewards > best_rewards:
                                        best_rewards = eval_rewards
                                        ddqn.save_models(mode=env_type)

                                        _, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='HER')
                                        logger.warning('best num step: {}'.format(test_env.num_steps))
                                        stats = test_env.view()
                                        stats.final_reward = eval_rewards
                                        stats.save(sub_dir = output_dir, plot = False)
                                break

                            ddqn.store_transition(transition[0], transition[1], transition[2],
                                                transition[3], False, new_goal)
                            ddqn.learn()
                            if ddqn.learn_step_counter != 0 and ddqn.learn_step_counter % result_saving_iter == 0:
                                eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='HER')
                                logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))
                                tracker.store(eval_rewards)

                                if eval_rewards > best_rewards:
                                    best_rewards = eval_rewards
                                    ddqn.save_models(mode=env_type)

                                    # _, test_env = self.evaluate_with_model(env=env, model=ddqn, type_reward='HER')
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
        # ddqn = ddqn.load_models(mode=env_type)
        # eval_rewards, test_env = self.evaluate_with_model(env=env, model=ddqn)
        # logger.success('Best Rewards: %s' % (round(eval_rewards, 2)))
        # stats = test_env.view()
        # stats.save(output_dir)
        # best_model.save_models(mode=env_type)
        # tools.plot_curve(x, num_steps, output_dir + 'step.png')