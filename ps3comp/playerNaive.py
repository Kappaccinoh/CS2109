import utils

class PlayerNaive:
    '''
    A naive agent that will always return the first available valid move.
    '''
    def make_move(self, board):
        return utils.generate_rand_move(board)
