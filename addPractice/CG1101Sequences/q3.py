def check_matrix(matrix):
    flat_list = [item for sublist in matrix for item in sublist]
    for i in range(len(flat_list) - 1):
        if flat_list[i] > flat_list[i + 1]:
            return False
    return True