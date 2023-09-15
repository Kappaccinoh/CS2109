SIZE = 9

board = (
(5, 3, 4, 6, 7, 8, 9, 1, 2),
(6, 0, 2, 1, 9, 0, 3, 4, 0),
(1, 9, 8, 3, 4, 2, 0, 6, 7),
(8, 5, 9, 7, 6, 1, 4, 2, 3),
(4, 2, 0, 8, 5, 3, 7, 9, 1),
(7, 1, 3, 9, 2, 4, 8, 5, 6),
(9, 6, 1, 0, 3, 7, 2, 8, 4),
(2, 8, 7, 4, 1, 9, 6, 0, 5),
(3, 4, 5, 2, 8, 6, 1, 7, 9))

def easy_sudoku(x, y, n):
    print(len(board))
    for i in range(1, SIZE + 1):
        if board[y][i] == n:
            return "Violation" 

    for i in range(1, SIZE + 1):
        if board[i][x] == n:
            return "Violation"

    return "No violation"
