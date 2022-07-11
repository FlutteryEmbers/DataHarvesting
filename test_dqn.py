from trainer.Q_Learning.ddqn import DDQN, MEMORY_CAPACITY
from environments.instances import discrete_single
from utils.utils import plot_curve
                   

n_games = 200
best_num_steps = 9999999999999999
episode_rewards = []
num_steps = []

env = discrete_single.TestEnvironment_2
ddqn = DDQN(env.get_linear_state_length(), 5, env)
ddqn.load_models(checkpoints=10)

done = False
s, current_position = env.reset()
episode_reward_sum = 0

while not done:
    a = ddqn.choose_action(s, current_position, disable_exploration=True)
    s_, r, done, current_position = env.step(a)

    ddqn.store_transition(s, a, r, s_, done)
    episode_reward_sum += r

    s = s_

stats = env.view()
stats.save()