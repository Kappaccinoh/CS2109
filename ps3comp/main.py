from negamax_alpha_beta import deepBlue

if __name__ == "__main__":
    print(deepBlue([['_', '_', '_', '_', '_', '_'], ['_', '_', 'B', 'B', '_', '_'], ['_', '_', '_', '_', 'B', 'B'], ['W', 'B', 'W', '_', 'B', '_'], ['_', '_', '_', '_', 'W', 'W'], ['_', 'W', 'W', '_', '_', '_']]))
    print(deepBlue([['_', '_', '_', '_', 'B', '_'], ['_', '_', '_', 'B', '_', '_'], ['_', '_', 'B', '_', '_', '_'], ['_', 'W', 'W', 'W', '_', '_'], ['_', '_', '_', '_', 'W', '_'], ['_', '_', '_', '_', '_', '_']]))
    print(deepBlue([['_', '_', '_', '_', 'B', '_'], ['_', '_', 'B', 'B', '_', '_'], ['_', '_', '_', '_', '_', '_'], ['_', 'W', 'W', 'W', '_', '_'], ['_', '_', '_', '_', 'W', '_'], ['_', '_', '_', '_', '_', '_']]))
