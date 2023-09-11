import utils

Score = int | float
Move = tuple[tuple[int, int], tuple[int, int]]

def evaluate(board):
    bcount = 0
    wcount = 0
    for r, row in enumerate(board):
        for tile in row:
            if tile == 'B':
                if r == 5:
                    return utils.WIN
                bcount += 1
            elif tile == 'W':
                if r == 0:
                    return -utils.WIN
                wcount += 1
    if wcount == 0:
        return utils.WIN
    if bcount == 0:
        return -utils.WIN
    return bcount - wcount

def generate_valid_moves(board):
    '''
    Generates a list (or iterable, if you want to) containing
    all possible moves in a particular position for black.
    '''
    allMoves = []

    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == 'B':
                src = r, c
                moveArray = [[1,-1], [1,0], [1,1]]
                for move in moveArray:
                    dr = src[0] + move[0]
                    dc = src[1] + move[1]
                    destination = dr, dc
                    if utils.is_valid_move(board, src, destination):
                        moveTuple = src, destination
                        allMoves.append(moveTuple)

    return allMoves


def test_11():
    board1 = [
        ['_', '_', 'B', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert sorted(generate_valid_moves(board1)) == [((0, 2), (1, 1)), ((0, 2), (1, 2)), ((0, 2), (1, 3))]

    board2 = [
        ['_', '_', '_', '_', 'B', '_'],
        ['_', '_', '_', '_', '_', 'B'],
        ['_', 'W', '_', '_', '_', '_'],
        ['_', '_', 'W', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_']
    ]
    assert sorted(generate_valid_moves(board2)) == [((0, 4), (1, 3)), ((0, 4), (1, 4)), ((1, 5), (2, 4)), ((1, 5), (2, 5))]

    board3 = [
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_'],
        ['_', 'W', 'W', 'W', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert sorted(generate_valid_moves(board3)) == [((1, 2), (2, 1)), ((1, 2), (2, 3))]

# test_11()

def minimax(board, depth, max_depth, is_black: bool) -> tuple[Score, Move]:
    '''
    Finds the best move for the input board state.
    Note that you are black.

    Parameters
    ----------
    board: 2D list of lists. Contains characters 'B', 'W' and '_'
    representing black pawn, white pawn and empty cell, respectively.

    depth: int, the depth to search for the best move. When this is equal
    to `max_depth`, you should get the evaluation of the position using
    the provided heuristic function.

    max_depth: int, the maximum depth for cutoff.

    is_black: bool. True when finding the best move for black, False
    otherwise.

    Returns
    -------
    A tuple (evalutation, ((src_row, src_col), (dst_row, dst_col))):
    evaluation: the best score that black can achieve after this move.
    src_row, src_col: position of the pawn to move.
    dst_row, dst_col: position to move the pawn to.
    '''
    v = minimaxCode(board, depth, max_depth, True)
    allMoves = generate_valid_moves(board)
    for move in allMoves:
        newBoard = utils.state_change(board, move[0], move[1], False)
        if (minimaxCode(newBoard, depth + 1, max_depth, False) == v):
            return (v, move)

def minimaxCode(board, depth, max_depth, is_black: bool):
    if depth > max_depth or utils.is_game_over(board):
        return evaluate(board)

    # recursion
    if not is_black:
        currBoard = utils.invert_board(board, False)
    else:
        currBoard = board

    allMoves = generate_valid_moves(currBoard)

    if is_black:
        bestMove = -1 * utils.WIN
        for move in allMoves:
            newBoard = utils.state_change(currBoard, move[0], move[1], False)
            newMove = minimaxCode(newBoard, depth + 1, max_depth, False)
            if (newMove > bestMove):
                bestMove = newMove

    else:
        bestMove = utils.WIN
        for move in allMoves:
            newBoard = utils.state_change(currBoard, move[0], move[1], False)
            newBoard = utils.invert_board(newBoard, False)
            newMove = minimaxCode(newBoard, depth + 1, max_depth, True)
            if (newMove < bestMove):
                bestMove = newMove

    return bestMove

            
def test_21():
    board1 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("___WB_"),
        list("_B__WW"),
        list("_WW___"),
    ]
    score1, _ = minimax(board1, 0, 1, True)
    assert score1 == utils.WIN, "black should win in 1"

    board2 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("_BW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score2, _ = minimax(board2, 0, 3, True)
    assert score2 == utils.WIN, "black should win in 3"

    board3 = [
        list("______"),
        list("__B___"),
        list("_WWW__"),
        list("______"),
        list("______"),
        list("______"),
    ]
    score3, _ = minimax(board3, 0, 4, True)
    assert score3 == -utils.WIN, "white should win in 4"

# test_21()

def negamax(board, depth, max_depth) -> tuple[Score, Move]:
    '''
    Finds the best move for the input board state.
    Note that you are black.

    Parameters
    ----------
    board: 2D list of lists. Contains characters 'B', 'W' and '_'
    representing black pawn, white pawn and empty cell, respectively.

    depth: int, the depth to search for the best move. When this is equal
    to `max_depth`, you should get the evaluation of the position using
    the provided heuristic function.

    max_depth: int, the maximum depth for cutoff.

    Notice that you no longer need the parameter `is_black`.

    Returns
    -------
    A tuple (evalutation, ((src_row, src_col), (dst_row, dst_col))):
    evaluation: the best score that black can achieve after this move.
    src_row, src_col: position of the pawn to move.
    dst_row, dst_col: position to move the pawn to.
    '''
    v = negamaxCode(board, depth, max_depth)
    allMoves = generate_valid_moves(board)
    for move in allMoves:
        newBoard = utils.state_change(board, move[0], move[1], False)
        if (negamaxCode(newBoard, depth + 1, max_depth) == v):
            return (v, move)

def negamaxCode(board, depth, max_depth):
    if depth > max_depth or utils.is_game_over(board):
        return evaluate(board)

    allMoves = generate_valid_moves(board)

    bestValue = -1 * utils.WIN
    for move in allMoves:
        newBoard = utils.state_change(board, move[0], move[1], False)
        flippedBoard = utils.invert_board(newBoard, False)
        value = -1 * negamaxCode(flippedBoard, depth + 1, max_depth)
        if value > bestValue:
            bestValue = value

    return bestValue

def test_22():
    board1 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("___WB_"),
        list("_B__WW"),
        list("_WW___"),
    ]
    score1, _ = negamax(board1, 0, 1)
    assert score1 == utils.WIN, "black should win in 1"

    board2 = [
        list("______"),
        list("___B__"),
        list("____BB"),
        list("_BW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score2, _ = negamax(board2, 0, 3)
    assert score2 == utils.WIN, "black should win in 3"

    board3 = [
        list("______"),
        list("__B___"),
        list("_WWW__"),
        list("______"),
        list("______"),
        list("______"),
    ]
    score3, _ = negamax(board3, 0, 4)
    assert score3 == -utils.WIN, "white should win in 4"

# test_22()

def minimax_alpha_beta(board, depth, max_depth, alpha, beta, is_black: bool) -> tuple[Score, Move]:
    # TODO: Replace this with your own implementation
    raise NotImplementedError

def test_31():
    board1 = [
        list("______"),
        list("__BB__"),
        list("____BB"),
        list("WBW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score1, _ = minimax_alpha_beta(board1, 0, 3, -utils.INF, utils.INF, True)
    assert score1 == utils.WIN, "black should win in 3"

    board2 = [
        list("____B_"),
        list("___B__"),
        list("__B___"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score2, _ = minimax_alpha_beta(board2, 0, 5, -utils.INF, utils.INF, True)
    assert score2 == utils.WIN, "black should win in 5"

    board3 = [
        list("____B_"),
        list("__BB__"),
        list("______"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score3, _ = minimax_alpha_beta(board3, 0, 6, -utils.INF, utils.INF, True)
    assert score3 == -utils.WIN, "white should win in 6"

# test_31()

def negamax_alpha_beta(board, depth, max_depth, alpha, beta) -> tuple[Score, Move]:
    # TODO: Replace this with your own implementation
    raise NotImplementedError

def test_32():
    board1 = [
        list("______"),
        list("__BB__"),
        list("____BB"),
        list("WBW_B_"),
        list("____WW"),
        list("_WW___"),
    ]
    score1, _ = negamax_alpha_beta(board1, 0, 3, -utils.INF, utils.INF)
    assert score1 == utils.WIN, "black should win in 3"

    board2 = [
        list("____B_"),
        list("___B__"),
        list("__B___"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score2, _ = negamax_alpha_beta(board2, 0, 5, -utils.INF, utils.INF)
    assert score2 == utils.WIN, "black should win in 5"

    board3 = [
        list("____B_"),
        list("__BB__"),
        list("______"),
        list("_WWW__"),
        list("____W_"),
        list("______"),
    ]
    score3, _ = negamax_alpha_beta(board3, 0, 6, -utils.INF, utils.INF)
    assert score3 == -utils.WIN, "white should win in 6"

# test_32()

# Uncomment and implement the function.
# Note: this will override the provided `evaluate` function.

# def evaluate(board):
#     # TODO: replace this with your own implementation
#     raise NotImplementedError

def test_41():
    board1 = [
        ['_', '_', '_', 'B', '_', '_'],
        ['_', '_', '_', 'W', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert evaluate(board1) == 0

    board2 = [
        ['_', '_', '_', 'B', 'W', '_'],
        ['_', '_', '_', 'W', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert evaluate(board2) == -utils.WIN

    board3 = [
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', 'B', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_'],
        ['_', '_', '_', '_', '_', '_']
    ]
    assert evaluate(board3) == utils.WIN

# test_41()

class PlayerAI:

    def make_move(self, board) -> Move:
        '''
        This is the function that will be called from main.py
        You should combine the functions in the previous tasks
        to implement this function.

        Parameters
        ----------
        self: object instance itself, passed in automatically by Python.
        
        board: 2D list-of-lists. Contains characters 'B', 'W', and '_' 
        representing black pawn, white pawn and empty cell respectively.
        
        Returns
        -------
        Two tuples of coordinates [row_index, col_index].
        The first tuple contains the source position of the black pawn
        to be moved, the second list contains the destination position.
        '''
        # TODO: Replace starter code with your AI
        ################
        # Starter code #
        ################
        for r in range(len(board)):
            for c in range(len(board[r])):
                # check if B can move forward directly
                if board[r][c] == 'B' and board[r + 1][c] == '_':
                    src = r, c
                    dst = r + 1, c
                    return src, dst # valid move
        return (0, 0), (0, 0) # invalid move

class PlayerNaive:
    '''
    A naive agent that will always return the first available valid move.
    '''
    def make_move(self, board):
        return utils.generate_rand_move(board)

##########################
# Game playing framework #
##########################
if __name__ == "__main__":
    assert utils.test_move([
        ['B', 'B', 'B', 'B', 'B', 'B'], 
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', '_', '_', '_', '_', '_'], 
        ['_', 'B', '_', '_', '_', '_'], 
        ['_', 'W', 'W', 'W', 'W', 'W'], 
        ['W', 'W', 'W', 'W', 'W', 'W']
    ], PlayerAI())

    assert utils.test_move([
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', '_', '_', '_', '_', '_'], 
        ['_', 'B', '_', '_', '_', '_'], 
        ['W', 'W', 'W', 'W', 'W', 'W'], 
        ['_', '_', 'W', 'W', 'W', 'W']
    ], PlayerAI())

    assert utils.test_move([
        ['_', '_', 'B', 'B', 'B', 'B'], 
        ['_', 'B', 'B', 'B', 'B', 'B'], 
        ['_', '_', '_', '_', '_', '_'], 
        ['_', 'B', 'W', '_', '_', '_'], 
        ['_', 'W', 'W', 'W', 'W', 'W'], 
        ['_', '_', '_', 'W', 'W', 'W']
    ], PlayerAI())

    # generates initial board
    board = utils.generate_init_state()
    res = utils.play(PlayerAI(), PlayerNaive(), board)
    # Black wins means your agent wins.
    # print(res)


if __name__ == "__main__":
    # 101010 for black
    print([['_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'B', '_', '_'], ['_', '_', '_', '_', 'B', 'B'], ['_', '_', '_', 'W', 'B', '_'], ['_', 'B', '_', '_', 'W', 'W'], ['_', 'W', 'W', '_', '_', '_']])
    print(negamax([['_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'B', '_', '_'], ['_', '_', '_', '_', 'B', 'B'], ['_', '_', '_', 'W', 'B', '_'], ['_', 'B', '_', '_', 'W', 'W'], ['_', 'W', 'W', '_', '_', '_']], 0, 1))

    # 101010 for black
    # print(negamax([['_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'B', '_', '_'], ['_', '_', '_', '_', 'B', 'B'], ['_', 'B', 'W', '_', 'B', '_'], ['_', '_', '_', '_', 'W', 'W'], ['_', 'W', 'W', '_', '_', '_']], 0, 3))

    # -101010 for black
    # print(negamax([['_', '_', '_', '_', '_', '_'], ['_', '_', 'B', '_', '_', '_'], ['_', 'W', 'W', 'W', '_', '_'], ['_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_'], ['_', '_', '_', '_', '_', '_']], 0 , 4))

    # test
    # print(minimax([['_', '_', '_', '_', '_', '_'], ['_', '_', '_', 'B', '_', '_'], ['_', '_', '_', '_', 'B', 'B'], ['_', '_', '_', 'W', 'B', '_'], ['_', '_', '_', '_', 'W', 'W'], ['_', 'W', 'B', '_', '_', '_']], 0, 1, True))