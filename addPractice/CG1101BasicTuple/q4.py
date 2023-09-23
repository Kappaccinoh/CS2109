from math import factorial
def pascal(n):
    # code to obtain a pascal triangle with n rows.
    res = ()
    for i in range(n):
        row = ()
        for j in range(i+1):
            ans = factorial(i)//(factorial(j)*factorial(i-j))
            row += (ans,)
    
        res += (row,)
    return res
    