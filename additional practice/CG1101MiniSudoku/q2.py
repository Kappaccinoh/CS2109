def fill_col(board):
    flag = False
    # write your code here
    for j in range(len(board)):
        size4set = {1,2,3,4}
        index = 0
        for i in range(len(board[0])):
            if board[i][j] in size4set:
                size4set.remove(board[i][j])
            else:
                index = i

        if len(size4set) == 1:
            board[index][j] = list(size4set)[0]
            flag = True

    # return True if any slot is filled, False otherwise
    return flag
