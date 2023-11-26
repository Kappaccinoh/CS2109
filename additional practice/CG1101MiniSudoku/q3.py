SIZE = 4
board1 = [[1, 0, 3, 0], [3, 0, 0, 2], [4, 3, 2, 1], [0, 0, 0, 3]]
board2 = [[0, 1, 3, 2], [2, 0, 1, 0], [1, 0, 0, 3], [3, 4, 2, 1]]
board3 = [[0, 0, 0, 0], [0, 1, 2, 4], [0, 3, 4, 1], [0, 4, 0, 2]]

'''
[0,1,3,2]
[2,0,1,4]
[1,2,4,3]
[3,4,2,1]
'''

def fill_section(board):
    flag = False

    # write your code here
    window = [
        [0,0],
        [1,0],
        [0,1],
        [1,1]
    ]

    # write your code here
    for i in range(0, len(board), 2):
        for j in range(0, len(board[0]), 2):
            size4set = {1,2,3,4}
            pos = 0
            for direc in window:
                if board[i + direc[0]][j + direc[1]] in size4set:
                    size4set.remove(board[i + direc[0]][j + direc[1]])
                else:
                    pos = (i + direc[0], j + direc[1])

            if len(size4set) == 1:
                board[pos[0]][pos[1]] = list(size4set)[0]
                flag = True

    # return True if any slot is filled, False otherwise
    return flag


if __name__ == "__main__":
    print(fill_section(board1))
    