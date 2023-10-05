def nth_digit(n, num):
    if n == 1:
        return int(str(num)[-1])
    else:
        return nth_digit(n - 1, num // 10)    
    
def mth_digit(m, num):
    if m == 1:
        return int(str(num)[0])
    else:
        print(num)
        return mth_digit(m - 1, int(str(num)[1:]))