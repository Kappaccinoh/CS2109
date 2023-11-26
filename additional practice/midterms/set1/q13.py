def count_change(amount, kinds_of_coins):
    def count_ways(amount, coin_index):
        # Base cases:
        if amount == 0:  # If amount becomes 0, one way is found.
            return 1
        if amount < 0 or coin_index < 0:  # If amount goes negative or no more coin types are available, no way is possible.
            return 0
        
        # Calculate the number of ways by including the current coin and excluding it
        ways_with_current_coin = count_ways(amount - coins[coin_index], coin_index)
        ways_without_current_coin = count_ways(amount, coin_index - 1)
        
        # Add both cases to get the total number of ways
        return ways_with_current_coin + ways_without_current_coin
    
    # Define the available coin denominations
    coins = [1, 5, 10, 20, 50, 100]  # Coin denominations in cents
    
    # Ensure kinds_of_coins is within the valid range
    if kinds_of_coins < 1 or kinds_of_coins > len(coins):
        return 0
    
    # Calculate the number of ways to make change using the specified kinds_of_coins
    return count_ways(amount, kinds_of_coins - 1)