from statistics import mode
from environments.runner import DDQN_GameAgent

agent = DDQN_GameAgent()
# agent.run(mode='DR')
agent.train(mode='Default', n_games=1)