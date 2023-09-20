import negamax_alpha_beta
# import ids
import zobrist_hashing

Score = int | float
Move = tuple[tuple[int, int], tuple[int, int]]

class PlayerAI:

    transpositionTable = dict()
    zobristTable = zobrist_hashing.initTable()

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