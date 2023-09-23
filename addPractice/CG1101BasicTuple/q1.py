import math
def get_hundredth(a, b, c):
    # code to get the digit in the hundredth position
    return (hundreds(a), hundreds(b), hundreds(c))

def hundreds(num):
    if num >= 0 and num < 100:
        return 0
    elif num < 0:
        return None
    return math.floor(( num / 100 ) % 10)
