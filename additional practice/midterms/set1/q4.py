def nth_digit(n, num):
    for i in range(n - 1):
        num = num // 10
    return int(str(num)[-1])
    

    
def mth_digit(m, num):
    num = str(num)
    for i in range(m - 1):
        num = num[1:]
    return int(num[0])