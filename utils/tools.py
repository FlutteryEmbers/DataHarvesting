import time, os, random, yaml, sys, re
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from loguru import logger

def set_logger_level(level):
    choice = ['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']
    logger.remove()
    # logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level=choice[level])
    logger.add(sys.stderr, level=choice[level])

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_curve(x, y, figure_file):
    plt.figure()
    plt.plot(x, y)
    # plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.close('all')

def mkdir(dir):
        isExist = os.path.exists(dir)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(dir)
            # print("The new directory is created!")
def setup_seed(seed):
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    T.backends.cudnn.deterministic = True

def load_config(file):
    logger.debug('loading {}'.format(file))
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(file, 'r') as stream:
        config = yaml.load(stream, Loader=loader)

    if config == None:
        sys.exit('{} did not loaded correctly'.format(file))
        
    return config

def save_network_params(checkpoint_file, state_dict, mode = 'Default'):
    if mode == 'Default':
        checkpoint_file = checkpoint_file
        logger.debug('saving default to {}'.format(checkpoint_file))
    else:
        checkpoint_file = checkpoint_file + '_' + mode
        logger.debug('saving {} to {}'.format(mode, checkpoint_file))

    T.save(state_dict, checkpoint_file)

def load_network_params(checkpoint_file, mode = 'Default'):
    if mode == 'Default':
        checkpoint_file = checkpoint_file
        logger.debug('loading default from {}'.format(checkpoint_file))
    else:
        checkpoint_file = checkpoint_file + '_' + mode
        logger.debug('loading from {}'.format(checkpoint_file))
    
    return T.load(checkpoint_file)

class dict2class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")