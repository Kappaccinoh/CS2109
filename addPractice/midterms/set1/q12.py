def replace_digit(n, d, r):
    res = [str(x) for x in str(n)]
    for i in range(len(res)):
        if res[i] == str(d):
            res[i] = str(r)
    
    return int("".join(res))