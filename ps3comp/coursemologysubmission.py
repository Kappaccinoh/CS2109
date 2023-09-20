import utils
import time
import sys
import random

MAX_DEPTH = sys.maxsize
ALPHA = -sys.maxsize
BETA = sys.maxsize
TIME_LIMIT = 1

NUM_PIECES = 2
EMPTY = 0
PAWN = 1 # black or white
BOARD_SIZE = 36 # 6x6
BOARD_LENGTH = 6

'''
Usage
zobristTable = initTable()
hashValue = computeHash(board, zobristTable) // For init only

newHashValue = updateHashOnMove(zobristTable, prevHashValue, src, dst, piece) // piece == '_' or piece == 'W' or piece == 'B'

'''

Score = int | float
Move = tuple[tuple[int, int], tuple[int, int]]

class PlayerAI:

    transpositionTable = dict()
    zobristTable = [[[random.randint(0, pow(2, 64)) for k in range(NUM_PIECES)] for j in range(BOARD_LENGTH)] for i in range(BOARD_LENGTH)]

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
        arr = deepBlue(board, self.transpositionTable, self.zobristTable)
        move = arr[0]
        self.transpositionTable = arr[1]
        self.zobristTable = arr[2]
        return move
        

def deepBlue(board, transpositionTable, zobristTable):
    hashValue = computeHash(board, zobristTable) # For init only

    start = time.time()
    bestMove = None
    max_depth = 1
    while (time.time() - start < TIME_LIMIT):
        currMove = negamax_alpha_beta(board, 0, max_depth, ALPHA, BETA, transpositionTable, zobristTable, hashValue)
        if currMove != None:
            bestMove = currMove
        max_depth += 1
    return (bestMove, transpositionTable, zobristTable)

def negamax_alpha_beta(board, depth, max_depth, alpha, beta, transpositionTable, zobristTable, hashValue) -> tuple[Score, Move]:
    v = negamax_alpha_betaCode(board, depth, max_depth, alpha, beta, transpositionTable, zobristTable, hashValue)
    allMoves = generate_valid_moves(board)
    for move in allMoves:

        # Main Code
        newBoard = utils.state_change(board, move[0], move[1], False)
        utils.invert_board(newBoard, True)

        # Updating TranspositionTable
        newHashValue = updateHashOnMove(zobristTable, hashValue, move[0], move[1], 'B')
        if newHashValue in transpositionTable:
            valuePair = transpositionTable[newHashValue]
            # valuePair (score, depth)
            if valuePair[1] < depth:
                # Update Transposition Table
                transpositionTable[newHashValue] = (evaluate(newBoard), depth)
        else:
            transpositionTable[newHashValue] = (evaluate(newBoard), depth)
        
        if (-negamax_alpha_betaCode(newBoard, depth + 1, max_depth, -beta, -alpha, transpositionTable, zobristTable, newHashValue) == v):
            return move
    # return ((0,0), (0,0))

def negamax_alpha_betaCode(board, depth, max_depth, alpha, beta, transpositionTable, zobristTable, currBoardsHashValue):
    # check transpositionTable for existing positions
    if currBoardsHashValue in transpositionTable:
        valuePair = transpositionTable[currBoardsHashValue]
        # valuePair (score, depth)
        if valuePair[1] >= depth:
            return valuePair[0]

    if depth >= max_depth or utils.is_game_over(board):
        return evaluate(board)
    
    allMoves = generate_valid_moves(board)
    bestValue = -1 * utils.WIN

    for move in allMoves:
        # Main Logic
        newBoard = utils.state_change(board, move[0], move[1], False)
        utils.invert_board(newBoard, True)

        # Updating TranspositionTable
        newHashValue = updateHashOnMove(zobristTable, currBoardsHashValue, move[0], move[1], 'B')
        if newHashValue in transpositionTable:
            valuePair = transpositionTable[newHashValue]
            # valuePair (score, depth)
            if valuePair[1] < depth:
                # Update Transposition Table
                transpositionTable[newHashValue] = (evaluate(newBoard), depth)
        else:
            transpositionTable[newHashValue] = (evaluate(newBoard), depth)

        # Main Logic Continued...
        value = -1 * negamax_alpha_betaCode(newBoard, depth + 1, max_depth, -beta, -alpha, transpositionTable, zobristTable, newHashValue)
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

# def randomInt():
#     min = 0
#     max = pow(2, 64)
#     return random.randint(min, max)

def indexOf(piece):
    if (piece=='_'):
        return 0
    else:
        return 1

# def initTable():
#     ZobristTable = [[[randomInt() for k in range(NUM_PIECES)] for j in range(BOARD_LENGTH)] for i in range(BOARD_LENGTH)]
#     return ZobristTable

def computeHash(board, ZobristTable):
    h = 0
    for i in range(BOARD_LENGTH):
        for j in range(BOARD_LENGTH):
            if (board[i][j] != '_'):
                piece = indexOf(board[i][j])
                h ^= ZobristTable[i][j][piece]
    return h

def updateHashOnMove(zobristTable, hashValue, src, dst, piece):
    hashValue ^= zobristTable[src[0]][src[1]][indexOf(piece)]
    hashValue ^= zobristTable[dst[0]][dst[1]][indexOf(piece)]
    return hashValue