def divisible_by_11(num):
    while num >= 0:
        num -= 11
        if num == 0:
            return True
    return False