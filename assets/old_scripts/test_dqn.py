from trainer.Q_Learning.ddqn import DDQN, MEMORY_CAPACITY
from environments import single_diescreteV1
from utils.tools import plot_curve
                   

n_games = 200
best_num_steps = 9999999999999999
episode_rewards = []
num_steps = []

env = single_diescreteV1.Test_Environment
ddqn = DDQN(inputs=len(env.status_tracker.get_state()), outputs=env.action_space.n, env=env)
ddqn.load_models(checkpoints=None, mode='DR')

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