# from negamax_alpha_beta import deepBlue
# import coursemologysubmission
# from coursemologysubmission import PlayerAI
import idscoursemology
from idscoursemology import PlayerAI
from playerNaive import PlayerNaive
import utils

if __name__ == "__main__":
    # print(deepBlue([['_', '_', '_', '_', '_', '_'], ['_', '_', 'B', 'B', '_', '_'], ['_', '_', '_', '_', 'B', 'B'], ['W', 'B', 'W', '_', 'B', '_'], ['_', '_', '_', '_', 'W', 'W'], ['_', 'W', 'W', '_', '_', '_']]))
    # print(deepBlue([['_', '_', '_', '_', 'B', '_'], ['_', '_', '_', 'B', '_', '_'], ['_', '_', 'B', '_', '_', '_'], ['_', 'W', 'W', 'W', '_', '_'], ['_', '_', '_', '_', 'W', '_'], ['_', '_', '_', '_', '_', '_']]))
    # print(deepBlue([['_', '_', '_', '_', 'B', '_'], ['_', '_', 'B', 'B', '_', '_'], ['_', '_', '_', '_', '_', '_'], ['_', 'W', 'W', 'W', '_', '_'], ['_', '_', '_', '_', 'W', '_'], ['_', '_', '_', '_', '_', '_']]))
    # board = [['_', '_', '_', '_', '_', '_'], ['_', '_', 'B', 'B', '_', '_'], ['_', '_', '_', '_', 'B', 'B'], ['W', 'B', 'W', '_', 'B', '_'], ['_', '_', '_', '_', 'W', 'W'], ['_', 'W', 'W', '_', '_', '_']]
    # board1 = [['_', '_', '_', '_', 'B', '_'], ['_', '_', '_', 'B', '_', '_'], ['_', '_', 'B', '_', '_', '_'], ['_', 'W', 'W', 'W', '_', '_'], ['_', '_', '_', '_', 'W', '_'], ['_', '_', '_', '_', '_', '_']]
    # board2 = [['_', '_', '_', '_', 'B', '_'], ['_', '_', 'B', 'B', '_', '_'], ['_', '_', '_', '_', '_', '_'], ['_', 'W', 'W', 'W', '_', '_'], ['_', '_', '_', '_', 'W', '_'], ['_', '_', '_', '_', '_', '_']]
    # currAI = PlayerAI()
    # print(currAI.make_move(board2))
    board = utils.generate_init_state()
    res = utils.play(PlayerAI(), PlayerNaive(), board)
    print(res)