import statistics as stat
def deviation(real_numbers):
    # code to calculate the standard deviation based on n real numbers
    return round(stat.pstdev(real_numbers),2)