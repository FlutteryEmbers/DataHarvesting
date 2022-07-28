import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import pickle, os

class Learning_Monitor():
	def __init__(self, output_dir='', name='', log = '') -> None:
		self.rewards = []
		self.name = name
		self.output_dir = output_dir + '/'
		self.log = log
		self.mkdir(output_dir)

	def store(self, reward):
		self.rewards.append(reward)
		logger.info('reward is {}'.format(reward))

	def average(self, n):
		reward = np.array(self.rewards)
		mean = np.mean(reward)
		logger.info('average rewards of last {} evaluation is {}'.format(n, mean))
		return mean

	def plot_learning_curve(self):
		name = '{}_rewards'.format(self.name)
		filename = self.output_dir + name + '.png'

		x = np.arange(0, len(self.rewards))

		plt.plot(x, self.rewards)
		plt.title(label=name)
		plt.savefig(filename)
		logger.success('successfully create {}'.format(filename))
		plt.close()

	def plot_average_learning_curve(self, n):
		name = '{}_average_{}_rewards'.format(self.name, n)
		filename = self.output_dir + name + '.png'

		reward = np.array(self.rewards)
		means = np.zeros(len(self.rewards))
		x = np.arange(len(self.rewards))
		for i in range(len(means)):
			means[i] = np.mean(reward[max(0, i-n):(i+1)])

		plt.plot(x, means)
		plt.title(label=name)
		plt.savefig(filename)
		logger.success('successfully create {}'.format(filename))
		plt.close()
	
	def dump_to_file(self):
		filename = self.output_dir + self.name + '_'
		with open(filename + 'history_rewards.pickle', 'wb') as handle:
			pickle.dump(self.rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

		handle.close()

	def load_from_file(self, file):
		with open(file, 'rb') as handle:
			self.rewards = pickle.load(handle)
		handle.close()

	def save_log(self):
		with open('readme.txt', 'w') as f:
			for line in self.log:
				f.write(line)
				f.write('\n')

		f.close()

	def reset(self):
		self.rewards = []

	def mkdir(self, dir):
		isExist = os.path.exists(dir)
		if not isExist:
            # Create a new directory because it does not exist 
			os.makedirs(dir)