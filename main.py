from statistics import mode
from environments.runner import DDQN_GameAgent

agent = DDQN_GameAgent()
# agent.run(mode='DR')
# agent.train(mode='Default', n_games=300)
# agent.run(mode='Default')

# agent.train(mode='DR', n_games=1000)
agent.run(mode='DR')