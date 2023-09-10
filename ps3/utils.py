import os
import sys
import copy
import time
import traceback
import multiprocessing
import timeout_decorator

# board row and column -> these are constant
ROW, COL = 6, 6
INF = 696969
WIN = 101010
MOVE_NONE = (-1, -1), (-1, -1)
TIME_LIMIT = 3.05

# generates initial state
def generate_init_state():
    state = [
        list("BBBBBB"),
        list("BBBBBB"),
        list("______"),
        list("______"),
        list("WWWWWW"),
        list("WWWWWW"),
    ]
    return state

# prints board
def print_state(board):
    horizontal_rule = '+' + ('-' * 5 + '+') * COL
    for row in board:
        print(horizontal_rule)
        print(f"|  {'  |  '.join(' ' if tile == '_' else tile for tile in row)}  |")
    print(horizontal_rule)

# inverts board by modifying board state, or returning a new board with updated board state
def invert_board(board, in_place=True):
    ''' 
    Inverts the board by modifying existing values if in_place is set to True, 
    or creating a new board with updated values if in_place is set to False.
    '''
    if not in_place:
        board = copy.deepcopy(board)
    board.reverse()
    for r, row in enumerate(board):
        for c, tile in enumerate(row):
            if tile == 'W':
                board[r][c] = 'B'
            elif tile == 'B':
                board[r][c] = 'W'
    return board

# checks if a move made for black is valid or not. Move source: src (row, col), move destination: dst (row, col)
def is_valid_move(board, src, dst):
    sr, sc = src
    dr, dc = dst
    if board[sr][sc] != 'B': # if move not made for black
        return False
    if dr < 0 or dr >= ROW or dc < 0 or dc >= COL: # if move takes pawn outside the board
        return False
    if dr != sr + 1: # if move takes more than one step forward
        return False
    if dc > sc + 1 or dc < sc - 1: # if move takes beyond left/right diagonal
        return False
    if dc == sc and board[dr][dc] != '_': # if pawn to the front, but still move forward
        return False
    if (dc == sc + 1 or dc == sc - 1) and board[dr][dc] == 'B': # if black pawn to the diagonal or front, but still move forward
        return False
    return True

# generates the first available valid move for black
def generate_rand_move(board):
    for r, row in enumerate(board):
        for c, tile in enumerate(row):
            if tile != 'B':
                continue
            src = r, c
            for d in (-1, 0, 1):
                dst = r + 1, c + d
                if is_valid_move(board, src, dst):
                    return src, dst
    raise ValueError("no valid move")

# makes a move effective on the board by modifying board state, or returning a new board with updated board state
def state_change(board, src, dst, in_place=True):
    ''' 
    Updates the board configuration by modifying existing values if in_place is set to True,
    or creating a new board with updated values if in_place is set to False.
    '''
    if not in_place:
        board = copy.deepcopy(board)
    if is_valid_move(board, src, dst):
        sr, sc = src
        dr, dc = dst
        board[sr][sc] = '_'
        board[dr][dc] = 'B'
    return board

# checks if game is over
def is_game_over(board):
    ''' 
    Returns True if game is over.
    '''
    if any(tile == 'B' for tile in board[5]) or any(tile == 'W' for tile in board[0]):
        return True
    wcount, bcount = 0, 0
    for row in board:
        for tile in row:
            if tile == 'B':
                bcount += 1
            elif tile == 'W':
                wcount += 1
    return bcount == 0 or wcount == 0


#############################################
# Utils function for game playing framework #
#############################################

# move making function for game playing
def make_move_job_func(player, board, queue):
    # disable stdout and stderr to prevent prints
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:
        src, dst = player.make_move(board)
        queue.put((src, dst))
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        queue.put(e)
        exit(1)
    finally:
        # reenable stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    return

# game playing function. Takes in the initial board
def play(playerAI_A, playerAI_B, board):
    colors = (black, white) = 'Black(Student agent)', 'White(Test agent)'
    players = []
    random_moves = 0
    move = 0

    # disable stdout for people who leave print statements in their code, disable stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:
        players.append(playerAI_A)
    except KeyboardInterrupt:
        exit()
    except:
        return f"{black} failed to initialise"
    finally:
        # reenable stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # disable stdout for people who leave print statements in their code, disable stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:
        players.append(playerAI_B)
    except KeyboardInterrupt:
        exit()
    except:
        return f"{white} failed to initialise"
    finally:
        # reenable stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # game starts
    color = None
    while not is_game_over(board):
        player = players[move % 2]
        color = colors[move % 2]
        src, dst = MOVE_NONE
        if color == white:
            invert_board(board)
            src, dst = player.make_move(board)
        else: # black
            result_queue = multiprocessing.Queue()
            board_copy = copy.deepcopy(board)
            mp = multiprocessing.Process(target=make_move_job_func, args=(player, board_copy, result_queue))
            mp.start()
            mp.join(timeout=3)
            exit_code = mp.exitcode
            if mp.is_alive():
                mp.terminate()
            if exit_code == None:
                del result_queue
            elif exit_code == 1:
                e = result_queue.get()
                del result_queue
                return f"{black} returned err={e} during move"
            elif exit_code == 0:
                src, dst = result_queue.get()
                del result_queue
            else:
                del result_queue
            
            is_valid = False
            try:
                is_valid = is_valid_move(board, src, dst)
            except KeyboardInterrupt:
                exit()
            except Exception:
                is_valid = False
            if not is_valid: # if move is invalid or time is exceeded, then we give a random move
                random_moves += 1
                src, dst = generate_rand_move(board)
        
        state_change(board, src, dst) # makes the move effective on the board
        if color == white:
            invert_board(board)
        move += 1

    # return f"{color} win; Random move made by {BLACK}: {random_moves};"
    return color == black

def wrap_test(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            return f'FAILED, reason: {e}'
    return inner

@wrap_test
@timeout_decorator.timeout(TIME_LIMIT)
def test_move(board, playerAI):
    board_copy = copy.deepcopy(board)
    start = time.time()
    src, dst = playerAI.make_move(board_copy)
    end = time.time()
    move_time = end - start
    valid = is_valid_move(board, src, dst)
    return valid and move_time <= 3
