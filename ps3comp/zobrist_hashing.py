import random

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

def randomInt():
    min = 0
    max = pow(2, 64)
    return random.randint(min, max)

def indexOf(piece):
    if (piece=='_'):
        return 0
    else:
        return 1

def initTable():
    ZobristTable = [[[randomInt() for k in range(NUM_PIECES)] for j in range(BOARD_LENGTH)] for i in range(BOARD_LENGTH)]
    return ZobristTable

def computeHash(board, ZobristTable):
    h = 0
    for i in range(8):
        for j in range(8):
            if (board[i][j] != '_'):
                piece = indexOf(board[i][j])
                h ^= ZobristTable[i][j][piece]
    return h

def updateHashOnMove(zobristTable, hash_value, src, dst, piece):
    hashValue ^= ZobristTable[src[0]][src[1]][indexOf(piece)]
    hashValue ^= ZobristTable[dst[0]][dst[1]][indexOf(piece)]
    return hash_value