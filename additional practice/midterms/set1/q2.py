def shift_right_one(num):
    num = str(num)
    last = num[-1]
    num = num[:-1]
    return int(last + num)

def shift_right(num, n):
    if n == 0:
        return num
    else:
        return shift_right(shift_right_one(num), n - 1)
    
    
def shift_right_alt(num, n):
    for i in range(n):
        num = shift_right_one(num)
    return num
