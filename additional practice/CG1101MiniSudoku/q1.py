SIZE = 4
board1 = [[1, 0, 3, 0], [3, 0, 0, 2], [4, 3, 2, 1], [0, 0, 0, 3]]
board2 = [[0, 1, 3, 2], [2, 0, 1, 0], [1, 0, 0, 3], [3, 4, 2, 1]]
board3 = [[0, 0, 0, 0], [0, 1, 2, 4], [0, 3, 4, 1], [0, 4, 0, 2]]

def fill_row(board):
    flag = False
    # write your code here
    for i in range(len(board)):
        size4set = {1,2,3,4}
        index = 0
        for j in range(len(board[0])):
            if board[i][j] in size4set:
                size4set.remove(board[i][j])
            else:
                index = j

        if len(size4set) == 1:
            board[i][index] = list(size4set)[0]
            flag = True

    # return True if any slot is filled, False otherwise
    return flag

if __name__ == "__main__":
    print(fill_row(board2))
    print(board2)
    

