from environments.instances.determistic import Test_Environment_Continuous, Test_Environment_Eval_Continuous
from utils import monitor, tools
from trainer.DDPG_HER import ddpg
from loguru import logger
import numpy as np

class GameAgent():
    def __init__(self) -> None:
        checkpoint_dir = 'model/her_ddpg'
        self.output_dir = 'results/HER'
        self.timer = tools.Timer()
        self.env = Test_Environment_Continuous
        self.evaluate_env = Test_Environment_Eval_Continuous

        self.state_dim = len(self.env.status_tracker.get_state())
        self.action_dim = self.env.action_space.shape
        self.max_action = float(self.env.action_space.high)
        self.max_episode_steps = self.env._max_episode_steps

        self.agent = ddpg.DDPGAgent(actor_learning_rate=1e-4, critic_learning_rate=1e-3, n_actions=2,
                                input_dims=self.state_dim, gamma=0.99,
                                memory_size=int(1e6), batch_size=64,
                                checkpoint_dir=checkpoint_dir)
    
    def evaluate_with_model(self, env, model):
        done = False
        s = env.reset()
        episode_reward_sum = 0
        goal = env.goal
        # print(goal)
        env.view()
        step = 0
        while not done and step < 1000:
            step += 1
            a = model.choose_action(s, goal, disable_exploration=False)
            s_, r, done, _ = env.step(a, type_reward='default')

            # model.store_transition(s, a, r, s_, done)
            episode_reward_sum += r
            
            s = s_

        return episode_reward_sum, env

    def train(self, n_games):
        best_num_steps = float('inf')
        num_steps = []

        output_dir = self.output_dir + '_train_ddpg/'

        tracker = monitor.Learning_Monitor(output_dir=output_dir, name='ddpg', log=['ddpg'])

        logger.warning('Using {} Environment'.format(self.env.status_tracker.name))
        self.env.state_mode = 'MLP'
        # env.mode = 'CNN'
        # ddqn = DDQN_CNN(env=env)
        best_model = None
        for i in range(n_games):
            logger.info('Start Episode: %s' % i)
            s = self.env.reset()
            # episode_reward_sum = 0

            self.timer.start()
            done = False
            transitions = []
            goal = self.env.goal
            logger.debug('current goal: {}'.format(goal))
            # for p in range(self.env._max_episode_steps):
            for p in range(10000):
                if not done:
                    # env.render()
                    a = self.agent.choose_action(s, goal, disable_exploration=False)
                    s_, r, done, _ = self.env.step(a, type_reward='default', verbose=1)

                    self.agent.store_experience(s, a, r, s_, done, goal)
                    transitions.append((s, a, r, s_))
                    # episode_reward_sum += r

                    s = s_

                    # if ddqn.memory_counter > ddqn.memory.mem_size:
                    self.agent.learn()

            if not done:
                print('NOT DONE')
                new_goal = np.copy(s)
                # logger.debug('alternative goal: {}'.format(new_goal))
                # print(new_goal)
                if not np.array_equal(new_goal, goal):
                    for p in range(self.env._max_episode_steps):
                        transition = transitions[p]
                        if np.array_equal(transition[3], new_goal):
                            self.agent.store_experience(transition[0], transition[1], 0.0,
                                                transition[3], True, new_goal)
                            self.agent.learn()
                            break

                        self.agent.store_experience(transition[0], transition[1], transition[2],
                                            transition[3], False, new_goal)
                        self.agent.learn()

            if i % 50 == 0 or n_games - i < 100:
                eval_rewards, test_env = self.evaluate_with_model(env=self.evaluate_env, model=self.agent)
                tracker.store(eval_rewards)
                logger.success('Episode %s Rewards: %s' % (i, round(eval_rewards, 2)))
                test_env.view()

            # ddqn.lr_decay(n_games)
            # episode_rewards.append(round(episode_reward_sum, 2))
            # num_steps.append(env.num_steps)
            

                if test_env.num_steps < best_num_steps:
                    logger.warning('best num step: {}'.format(test_env.num_steps))
                    best_num_steps = test_env.num_steps
                    best_model = self.agent
                    
            
            self.timer.stop()

        x = [i+1 for i in range(n_games)]
        # tools.plot_curve(x, episode_rewards, 'results/' + env_type + '/rewards.png')
        tracker.plot_average_learning_curve(50)
        tracker.plot_learning_curve()
        tracker.dump_to_file()
        tracker.save_log()

        eval_rewards, test_env = self.evaluate_with_model(env=self.evaluate_env, model=best_model)
        logger.success('Best Rewards: %s' % (round(eval_rewards, 2)))
        stats = test_env.view()
        stats.save(output_dir)
        best_model.save_models(mode='Default')

    def evaluate(self):
        pass

