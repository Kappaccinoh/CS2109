def replace_digit(n, d, r):
    # Base case: If n is 0, return 0
    if n == 0:
        return 0
    
    # Extract the last digit of n
    last_digit = n % 10
    
    # Recursively replace digits and build the result
    new_digit = last_digit if last_digit != d else r
    rest_of_number = n // 10
    
    # Combine the new_digit with the result of the recursive call
    result = replace_digit(rest_of_number, d, r) * 10 + new_digit
    
    return result
