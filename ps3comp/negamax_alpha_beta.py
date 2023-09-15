import zobrist_hashing
import utils
import time
import sys

MAX_DEPTH = sys.maxsize
ALPHA = -sys.maxsize
BETA = sys.maxsize
TIME_LIMIT = 2.5

Score = int | float
Move = tuple[tuple[int, int], tuple[int, int]]

# DeepBlue:
    # minimax (DONE)
    # alpha beta (DONE)
    # progressive deepening (DONE)
    # opening book (DONE)
    # endgame (DONE)
    # uneven tree development (too hard lol)
    # parallel computing?? (what)

    # Transposition Table (Zobrist Hashing) - Database essentially for Opening and Endgame

def deepBlue(board):
    # transpositionTable = dict()
    # zobristTable = initTable()
    # hashValue = computeHash(board, zobristTable) # For init only
    transpositionTable = 0
    zobristTable = 0

    start = time.time()
    bestMove = None
    max_depth = 1
    while (time.time() - start < TIME_LIMIT):
        bestMove = negamax_alpha_beta(board, 0, max_depth, ALPHA, BETA, transpositionTable, zobristTable)
        max_depth += 1
    return bestMove

def negamax_alpha_beta(board, depth, max_depth, alpha, beta, transpositionTable, zobristTable) -> tuple[Score, Move]:
    v = negamax_alpha_betaCode(board, depth, max_depth, alpha, beta, transpositionTable, zobristTable)
    allMoves = generate_valid_moves(board)
    for move in allMoves:
        newBoard = utils.state_change(board, move[0], move[1], False)
        utils.invert_board(newBoard, True)
        if (-negamax_alpha_betaCode(newBoard, depth + 1, max_depth, -beta, -alpha, transpositionTable, zobristTable) == v):
            return (v, move)

def negamax_alpha_betaCode(board, depth, max_depth, alpha, beta, transpositionTable, zobristTable):
    if depth >= max_depth or utils.is_game_over(board):
        return evaluate(board)
    
    allMoves = generate_valid_moves(board)
    bestValue = -1 * utils.WIN

    for move in allMoves:
        newBoard = utils.state_change(board, move[0], move[1], False)
        utils.invert_board(newBoard, True)
        value = -1 * negamax_alpha_betaCode(newBoard, depth + 1, max_depth, -beta, -alpha, transpositionTable, zobristTable)
        if value >= bestValue:
            bestValue = value
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return bestValue


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

def evaluate(board):
    blackPieces = 0
    whitePieces = 0

    boardLength = len(board)

    blackEvaluationBoard = [
        10,10,20,40,50,utils.WIN
    ]

    whiteEvaluationBoard = [
        -utils.WIN,-50,-40,-20,-10,-10
    ]

    blackEval = 0
    whiteEval = 0

    for i in range(boardLength):
        for j in range(boardLength):
            piece = board[i][j]
            if piece == 'B':
                blackPieces += 1
                blackEval += blackEvaluationBoard[i]
                if i == boardLength - 1:
                    return utils.WIN
            elif piece == 'W':
                whitePieces += 1
                whiteEval += whiteEvaluationBoard[i]
                if i == 0:
                    return -utils.WIN
            
    if blackPieces == 0 and whitePieces != 0:
        return -utils.WIN
    if whitePieces == 0 and blackPieces != 0:
        return utils.WIN
    
    score = blackEval + whiteEval
    return score
