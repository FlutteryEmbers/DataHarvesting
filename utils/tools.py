def ListAddition(L1, L2):
    next_position = []
    for (c1, c2) in zip(L1, L2):
        next_position.append(c1+c2)
    return next_position

def BlockPrint(title, item):
    print('=================' + title + "======================")
    print(item)
    print("====================================================")

def LabelPrint(title, item):
    print(title, end=': ')
    print(item)