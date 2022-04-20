class Environment():
    def __init__(self, reward_func):
        self.reward_func = reward_func
        
    def init(self, initial_x, initial_y, board):
        self.board = board
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.x = initial_x
        self.y = initial_y

    def reset(self):
        self.x = self.initial_x
        self.y = self.initial_y
    
    def step(self, action):
        pass

    def reward(self):
        pass
        return self.reward

    def visualizer(self):
        pass