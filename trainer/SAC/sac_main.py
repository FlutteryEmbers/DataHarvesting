from environments.instances.determistic import Test_Environment_Continuous, Test_Environment_Eval_Continuous
from utils import monitor, tools
from trainer.SAC.sac import SAC
from trainer.SAC.HERMemory import HindsightExperienceReplayMemory
from loguru import logger
import numpy as np

from utils.buffer import ReplayBuffer

class Agent():
    def __init__(self) -> None:
        checkpoint_dir = 'model/sac'
        self.output_dir = 'results/SAC'
        self.timer = tools.Timer()
        self.env = Test_Environment_Continuous
        self.evaluate_env = Test_Environment_Eval_Continuous

        self.state_dim = len(self.env.status_tracker.get_state())
        self.action_dim = self.env.action_space.shape
        self.max_action = float(self.env.action_space.high)
        self.max_episode_steps = self.env._max_episode_steps

        self.agent = SAC(self.state_dim, self.action_dim, self.max_action)

    def evaluate_with_model(self, env, model):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = model.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a, type_reward='default')
            episode_reward += r
            s = s_
        print(episode_reward)

        return episode_reward, env

    def train(self, max_train_steps):
        output_dir = self.output_dir + '_train_sac/'
        best_num_steps = float('inf')
        best_model = None

        max_train_steps = 3e6  # Maximum number of training steps
        random_steps = 25e3  # Take the random actions in the beginning for the better exploration
        evaluate_freq = 5e3  # Evaluate the policy every 'evaluate_freq' steps
        evaluate_num = 0  # Record the number of evaluations
        total_steps = 0  # Record the total steps during the training
        memory_size = int(1e6) ## Variable
        replay_buffer = HindsightExperienceReplayMemory(memory_size=memory_size, input_dims=self.state_dim, n_actions=self.action_dim)
        tracker = monitor.Learning_Monitor(output_dir=output_dir, name='sac', log=['ddpg'])

        while total_steps < max_train_steps:
            s = self.env.reset()
            episode_steps = 0
            done = False
            goal = self.env.goal
            while not done:
                episode_steps += 1
                if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                    a = self.env.action_space.sample()
                else:
                    a = self.agent.choose_action(s)
                s_, r, done, _ = self.env.step(a, type_reward='default')

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != self.max_episode_steps:
                    dw = True
                else:
                    dw = False
                    
                replay_buffer.add_experience(s, a, r, s_, dw, goal)  # Store the transition
                s = s_

                if total_steps >= random_steps:
                    self.agent.learn(replay_buffer)

                if (total_steps + 1) % evaluate_freq == 0:
                    evaluate_num += 1
                    evaluate_reward, test_env = self.evaluate_with_model(env=self.evaluate_env, model=self.agent)
                    print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
                    tracker.store(evaluate_reward)

                    if test_env.num_steps < best_num_steps:
                        logger.warning('best num step: {}'.format(test_env.num_steps))
                        best_num_steps = test_env.num_steps
                        best_model = self.agent

                total_steps += 1

        tracker.plot_average_learning_curve(50)
        tracker.plot_learning_curve()
        tracker.dump_to_file()
        tracker.save_log()

        eval_rewards, test_env = self.evaluate_with_model(env=self.evaluate_env, model=best_model)
        logger.success('Best Rewards: %s' % (round(eval_rewards, 2)))
        stats = test_env.view()
        stats.save(output_dir)
        # best_model.save_models(mode='Default')