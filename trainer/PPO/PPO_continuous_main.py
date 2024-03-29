import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from trainer.PPO.normalization import Normalization, RewardScaling
from trainer.PPO.replaybuffer import ReplayBuffer
from trainer.PPO.ppo_continuous import PPO_continuous
from environments.instances.determistic import Test_Environment_Continuous, Test_Environment_Eval_Continuous
from environments.instances.randomized import DR_Environment_Continuous, DR_Environment_Eval_Continuous
from utils import tools
from loguru import logger

class PPO_GameAgent():
    def __init__(self, args) -> None:
        self.args = args
        self.timer = tools.Timer()

    def train(self, env_type, env=Test_Environment_Continuous):
        self.main(args=self.args, env=env, env_type=env_type)

    def evaluate_policy(self, args, env=Test_Environment_Eval_Continuous, agent=None, state_norm=None, load_model=None):
        if load_model != None:
            logger.success('evaluation mode {}'.format(load_model))
            logger.success('environment name {}'.format(env.status_tracker.name))
            logger.success('action type {}'.format(env.action_type))
            args.env_type = load_model
            args.state_dim = len(env.status_tracker.get_state())
            args.action_dim = env.action_space.shape
            args.max_action = float(env.action_space.high)
            args.max_episode_steps = env._max_episode_steps
            args.use_orthogonal_init = False

            agent = PPO_continuous(args, load_model=load_model)
            state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
            if args.use_reward_norm:  # Trick 3:reward normalization
                reward_norm = Normalization(shape=1)
            elif args.use_reward_scaling:  # Trick 4:reward scaling
                reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        times = 3
        evaluate_reward = 0
        for _ in range(times):
            s = env.reset()
            if args.use_state_norm:
                s = state_norm(s, update=False)  # During the evaluating,update=False
            done = False
            episode_reward = 0
            while not done:
                a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                s_, r, done, position = env.step(action)
                # print(position)
                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += r
                s = s_
            evaluate_reward += episode_reward
            stats = env.view()
            if load_model != None:
                stats.save('{}/'.format(load_model))
            else:
                num_steps = env.num_steps
                if num_steps < self.best_num_steps:
                    self.best_num_steps = num_steps
                    agent.actor.save_checkpoint(mode=args.env_type)
                    agent.critic.save_checkpoint(mode=args.env_type)

        return evaluate_reward / times


    def main(self, args, env, env_type='Default'):
        self.total_eval = args.max_train_steps / args.evaluate_freq
        self.best_num_steps = float('inf')
        logger.success('total {} evals'.format(self.total_eval))

        env = Test_Environment_Continuous
        env_evaluate = Test_Environment_Eval_Continuous

        if env_type == 'DR':
            env = DR_Environment_Continuous
            env_evaluate = DR_Environment_Eval_Continuous

        logger.success('trainning:')
        logger.success(env.status_tracker.name)
        logger.success(env.action_type)

        args.env_type = env_type
        args.state_dim = len(env.status_tracker.get_state())
        args.action_dim = env.action_space.shape
        args.max_action = float(env.action_space.high)
        args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
        #TODO: print("env={}".format(env_name))
        logger.trace("state_dim={}".format(args.state_dim))
        logger.trace("action_dim={}".format(args.action_dim))
        logger.trace("max_action={}".format(args.max_action))
        logger.trace("max_episode_steps={}".format(args.max_episode_steps))

        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training

        replay_buffer = ReplayBuffer(args)
        agent = PPO_continuous(args)

        # Build a tensorboard
        # writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))
        
        state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if args.use_reward_norm:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif args.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        self.timer.start()
        while total_steps < args.max_train_steps:
            s = env.reset()
            if args.use_state_norm:
                s = state_norm(s)
            if args.use_reward_scaling:
                reward_scaling.reset()
            episode_steps = 0
            done = False
            while not done:
                episode_steps += 1
                a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                s_, r, done, _ = env.step(action)

                if args.use_state_norm:
                    s_ = state_norm(s_)
                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != args.max_episode_steps:
                    dw = True
                else:
                    dw = False

                # Take the 'action'，but store the original 'a'（especially for Beta）
                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                total_steps += 1

                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer.count == args.batch_size:
                    agent.update(replay_buffer, total_steps)
                    replay_buffer.count = 0

                # Evaluate the policy every 'evaluate_freq' steps
                if total_steps % args.evaluate_freq == 0:
                    self.timer.stop()
                    logger.success("evaluate_num:{} left: {}".format(evaluate_num, self.total_eval - evaluate_num))
                    evaluate_num += 1
                    evaluate_reward = self.evaluate_policy(args, env_evaluate, agent, state_norm)
                    evaluate_rewards.append(evaluate_reward)
                    logger.success("evaluate_reward:{}".format(evaluate_reward))
                    agent.actor.save_checkpoint(mode='tmp')
                    agent.critic.save_checkpoint(mode='tmp')
                    self.timer.start()

        self.timer.stop()
        # env_type = 'Default'
        

        x = [i+1 for i in range(len(evaluate_rewards))]
        tools.plot_curve(x, evaluate_rewards, 'results/' + env_type + '/rewards.png')
        # tools.plot_curve(x, num_steps, 'results/' + env_type + '/step.png')