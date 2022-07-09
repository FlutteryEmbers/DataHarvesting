from environments.config.single_discrete import DQN_Environment

board = [[3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0]]
startAt = [0 ,0]
arrivalAt = [4,4]

TestEnvironment = DQN_Environment(board=board)
data_volumn = [200, 500, 800]
TestEnvironment.init(startAt=startAt, arrivalAt=arrivalAt, data_volume=data_volumn)


board = []
for i in range(10):
    boardrow = []
    for j in range(10):
        boardrow.append(0)
    board.append(boardrow)
startAt = [0, 0]
arrivalAt = [9, 9]
board[0][1] = 1 ## first tower
board[4][7] = 2 ## second tower
board[9][3] = 3 ## third tower
# board[27][27] = 4
# board[13][12] = 5
# board[40][30] = 6
TestEnvironment_2 = DQN_Environment(board=board)
data_volumn = [300, 500, 800]
TestEnvironment_2.init(startAt=startAt, arrivalAt=arrivalAt, data_volume=data_volumn)