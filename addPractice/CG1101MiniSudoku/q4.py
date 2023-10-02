def fill_board(board):
    fillable = True  # indicate if a board is still fillable

    while(fillable):
    # if still fillable, you shall continue filling it
    # if no longer fillable, shall return the board
        if not fill_row(board) and not fill_col(board) and not fill_section(board):
            fillable = not fillable

    # write your code here
    return board
