import numpy as np
from torch.utils.tensorboard import SummaryWriter
from trainerV2.MA_PPO.scripts.normalization import Normalization, RewardScaling
from trainerV2.MA_PPO.scripts.replaybuffer import ReplayBuffer
from trainerV2.MA_PPO.scripts.ppo_continuous import PPO_continuous
from utils import tools, monitor
from loguru import logger
from datetime import datetime

class PPO_GameAgent():
    def __init__(self, args, output_dir, train_mode=True, debug = False) -> None:
        self.args = args
        self.timer = tools.Timer()
        self.eval_times = 0
        
        now = datetime.now()
        current_time = now.strftime("%b%d-%H_%M")
       
        self.output_dir = output_dir

        if train_mode and not debug:
            self.running_summary = SummaryWriter(log_dir='cache/runs/' + '{}-{}-{}'.format(current_time, args.run_name, args.seed))
        elif debug:
            self.running_summary = SummaryWriter(log_dir='cache/runs/' + 'debug')

        if train_mode:
            self.output_dir = output_dir + '{}-{}'.format(current_time, args.run_name) + '/'
            self.args.run_info = '{}_{}'.format(current_time, args.run_name)
            tools.mkdir(self.output_dir+'/model/')
            tools.mkdir(self.output_dir+'/logs/')

    def train(self, env):
        self.main(args=self.args, env=env)

    def evaluate(self, env):
        args = self.args
        args.state_dim = len(env.get_state())
        args.action_dim = env.action_space.shape
        args.max_action = float(env.action_space.high)
        args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
        # args.type_reward = 'Shaped_Reward'
        
        self.evaluate_policy(args=args, env=env, display=True, load_model=True)

    def evaluate_policy(self, args, env, agent=None, state_norm=None, load_model=None, display=False):
        if load_model != None:
            logger.success('evaluation mode {}'.format(load_model))
            # logger.success('environment name {}'.format(env.name))
            logger.success('action type {}'.format(env.action_type))
            args.env_type = load_model
            args.state_dim = len(env.get_state())
            args.action_dim = env.action_space.shape
            args.max_action = float(env.action_space.high)
            args.max_episode_steps = env._max_episode_steps
            args.use_orthogonal_init = False

            agent = PPO_continuous(args, load_model=load_model, chkpt_dir=self.output_dir + '/model/')
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
            while not done and env.num_steps < 200:
                a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                s_, r, done, position = env.step(action, args)
                # print(position)
                env.render(display=display)
                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += r
                s = s_
            evaluate_reward += episode_reward
            stats = env.view()
            if load_model != None:
                stats.save(sub_dir = self.output_dir, plot = True)
            else:
                self.eval_times += 1
                num_steps = env.num_steps
                if num_steps < self.best_num_steps:
                    self.best_num_steps = num_steps
                if evaluate_reward / times > self.best_reward:
                    tools.mkdir(self.output_dir+'/best_case/')
                    self.best_reward = evaluate_reward / times
                    stats.save(sub_dir = self.output_dir+'/best_case/', plot = True)
                    agent.save_models()

                if self.eval_times % 20 == 0:
                    tools.mkdir(self.output_dir+'/tmp_case/')
                    stats.save(sub_dir = self.output_dir+'/tmp_case/', plot = True)
        
        return evaluate_reward / times

    def evaluate_robust(self, env, dirs, noise_type=None, state_norm=None, display=False):
        args = self.args
        args.state_dim = len(env.get_state())
        args.action_dim = env.action_space.shape
        args.max_action = float(env.action_space.high)
        args.num_agents = env.agents.num_agents
        args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
        # args.type_reward = 'Shaped_Reward'

        logger.success('action type {}'.format(env.action_type))
        args.env_type = True
        args.state_dim = len(env.get_state())
        args.action_dim = env.action_space.shape
        args.max_action = float(env.action_space.high)
        args.max_episode_steps = env._max_episode_steps
        args.use_orthogonal_init = False

        agent = PPO_continuous(args, chkpt_dir=self.output_dir + '/model/')
        agent.actor.load_checkpoint(chkpt_dir=dirs['actor'])
        agent.critic.load_checkpoint(chkpt_dir=dirs['critic'])
        if noise_type == 'adv':
            agent.adv_net.load_checkpoint(chkpt_dir=dirs['adv_net'])

        state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if args.use_reward_norm:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif args.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        times = 1
        
        if noise_type == 'random':
            times = 10

        evaluate_reward = 0
        episode_rewards = []
        episode_steps = []
        for _ in range(times):
            s = env.reset()
            if args.use_state_norm:
                s = state_norm(s, update=False)  # During the evaluating,update=False
            done = False
            episode_reward = 0
            while not done and env.num_steps < 200:
                if noise_type == 'adv':
                    noise = agent.adv_net(s)
                    s = s + noise.cpu().detach().numpy()
                elif noise_type == 'random':
                    noise = np.random.rand(len(s)) * args.delta
                    s = s + noise
                a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                s_, r, done, position = env.step(action, args)
                # print(position)
                env.render(display=display)
                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += r
                s = s_
            stats = env.view()
            stats.save(sub_dir = self.output_dir, plot = True)
            evaluate_reward += episode_reward
            episode_rewards.append(episode_reward)
            episode_steps.append(env.num_steps)
           
        return evaluate_reward / times, np.var(np.array(episode_rewards)), np.mean(np.array(episode_steps)), np.var(np.array(episode_steps))

    def main(self, args, env):
        self.total_eval = args.max_train_steps / args.evaluate_freq
        self.best_num_steps = float('inf')
        self.best_reward = -float('inf')
        
        logger.success('total {} evals'.format(self.total_eval))
        logger.success('trainning:')
        # logger.success(env.status_tracker.name)
        logger.success(env.action_type)

        args.state_dim = len(env.get_state())
        args.action_dim = env.action_space.shape
        args.max_action = float(env.action_space.high)
        args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
        args.num_agents = env.board.agents.num_agents

        args.train_adv = False if args.train_adv == None else args.train_adv

        logger.trace("state_dim={}".format(args.state_dim))
        logger.trace("action_dim={}".format(args.action_dim))
        logger.trace("max_action={}".format(args.max_action))
        logger.trace("max_episode_steps={}".format(args.max_episode_steps))

        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training

        replay_buffer = ReplayBuffer(args)
        agent = PPO_continuous(args, chkpt_dir=self.output_dir + '/model/', train_adv=args.train_adv)

        # Build a tensorboard
        # writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))
        
        state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if args.use_reward_norm:  # Trick 3:reward normalization
            reward_norm = Normalization(shape=1)
        elif args.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

        self.learning_monitor = monitor.Learning_Monitor(output_dir=self.output_dir+'/logs/', name='ppo', args=args)
        self.learning_monitor.save_log()

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

                s_, r, done, _ = env.step(action, args)

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
                    logger.success("evaluate_num:{} left: {} - {}%".format(evaluate_num, self.total_eval - evaluate_num, (self.total_eval - evaluate_num)/self.total_eval*100))
                    evaluate_num += 1
                    evaluate_reward = self.evaluate_policy(args, env, agent, state_norm)
                    evaluate_rewards.append(evaluate_reward)
                    self.learning_monitor.store(evaluate_reward)
                    self.running_summary.add_scalar('info/rewards', evaluate_reward, total_steps)
                    self.running_summary.add_scalar('info/best_steps', self.best_num_steps, total_steps)
                    self.running_summary.add_scalar('info/best_rewards', self.best_reward, total_steps)
                    self.running_summary.add_scalar('info/average_rewards', self.learning_monitor.average(50), total_steps)

                    self.running_summary.add_scalar('adv_loss/adv_loss', agent.adv_loss, total_steps)
                    self.running_summary.add_histogram("layer1.bias", agent.adv_net.fc1.bias, total_steps)
                    self.running_summary.add_histogram("conv1.weight", agent.adv_net.fc1.weight, total_steps)
                    self.running_summary.add_histogram("conv2.bias", agent.adv_net.fc2.bias, total_steps)
                    self.running_summary.add_histogram("conv2.weight", agent.adv_net.fc2.weight, total_steps)
                    logger.success("evaluate_reward:{}".format(evaluate_reward))
                    # agent.actor.save_checkpoint(mode='tmp')
                    # agent.critic.save_checkpoint(mode='tmp')
                    self.timer.start()
        self.learning_monitor.plot_average_learning_curve(50)
        self.learning_monitor.plot_learning_curve()
        self.learning_monitor.dump_to_file()
        self.timer.stop()
        # env_type = 'Default'
        

        # x = [i+1 for i in range(len(evaluate_rewards))]
        # tools.plot_curve(x, evaluate_rewards, 'results/' + env_type + '/rewards.png')
        # tools.plot_curve(x, num_steps, 'results/' + env_type + '/step.png')