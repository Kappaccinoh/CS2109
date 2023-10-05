def divisible_by_11(num):
    if num == 0:
        return True
    elif num < 0:
        return False
    else:
        return divisible_by_11(num - 11) & True