from collections import deque
import copy

class node:
    def __init__(self, arr, direction, visited, action, backtrack):
        self.arr = arr
        self.direction = direction # boolean True for left, False for right
        self.visited = visited # boolean
        self.action = action
        self.backtrack = backtrack

# Task 1.6
def mnc_tree_search(m, c):
    '''
    Solution should be the action taken from the root node (initial state) to 
    the leaf node (goal state) in the search tree.

    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    
    # start = [[m,c],[0,0]]
    # goal = [[0,0],[m,c]]
    queue = deque()
    root = node([[m,c],[0,0]], True, False, -1, None)
    queue.append(root)
    while len(queue) != 0:
        curr = queue.popleft()

        # total possibilities of moving m's and c's from one side to another - 5
        for action in range(5):
            curr_array = copy.deepcopy(curr.arr)
            temp = node(curr_array, curr.direction, curr.visited, curr.action, curr.backtrack)
            next_state = transitionTo(temp, action, temp.direction)

            if not isValidState(next_state, m, c):
                continue # ignore and dont add to queue because state is invalid

            if next_state.arr == [[0,0],[m,c]]:
                solution = getSolution(next_state, [[m,c],[0,0]])
                solution = tuple(solution)
                return solution
            queue.append(next_state)
        
        curr.visited = True
    
    return False


# I define 5 possible choices that the boat can choose from when moving from one side to another
# 0: 1m
# 1: 1c
# 2: 1m1c
# 3: 2m
# 4: 2c

def getTransition(action):
    match action:
        case 0:
            return (1,0)
        case 1:
            return (0,1)
        case 2:
            return (1,1)
        case 3:
            return (2,0)
        case 4:
            return (0,2)

def transitionTo(state, action, LR):
    newState = node(state.arr, state.direction, state.visited, action, state.backtrack)
    newState.direction = not newState.direction
    newState.action = action
    newState.backtrack = state

    # LR is boolean, if True boat needs to go right, else left
    # "factor" determines which direction the boat is travelling in
    if LR:
        factor = 1
    else:
        factor = -1
    # factor = 1 if LR else factor = -1

    match action:
        case 0:
            # 1m
            newState.arr[0][0] += -1 * factor
            newState.arr[1][0] += 1 * factor
            return newState
        case 1:
            # 1c
            newState.arr[0][1] += -1 * factor
            newState.arr[1][1] += 1 * factor
            return newState
        case 2:
            # 1m1c
            newState.arr[0][0] += -1 * factor
            newState.arr[1][0] += 1 * factor
            newState.arr[0][1] += -1 * factor
            newState.arr[1][1] += 1 * factor
            return newState
        case 3:
            # 2m
            newState.arr[0][0] += -2 * factor
            newState.arr[1][0] += 2 * factor
            return newState
        case 4:
            #2c
            newState.arr[0][1] += -2 * factor
            newState.arr[1][1] += 2 * factor
            return newState

def isValidState(state, m, c):
    curr = state.arr
    # been visited
    if state.visited:
        return False

    # check if total number of cannibals and missionaries is less than m and c and more than 0
    if curr[0][0] < 0 or curr[0][0] > m or curr[1][0] < 0 or curr[1][0] > m:
        return False
    if curr[0][1] < 0 or curr[0][1] > c or curr[1][1] < 0 or curr[1][1] > c:
        return False

    # check if cannibals outnumber missionaries
    # side0
    if curr[0][0] < curr[0][1] and curr[0][0] != 0:
        return False

    if curr[1][0] < curr[1][1] and curr[1][0] != 0:
        return False
    
    return True

def getSolution(state, start):
    solution = []
    curr = state
    while (curr.arr != start):
        transitionTuple = getTransition(curr.action)
        solution.insert(0, transitionTuple)
        curr = curr.backtrack
        if curr == None:
            solution.pop(0)
            return solution

# Test cases for Task 1.6
def test_16():
    expected = ((2, 0), (1, 0), (1, 1))
    assert(mnc_tree_search(2,1) == expected)

    expected = ((1, 1), (1, 0), (2, 0), (1, 0), (1, 1))
    assert(mnc_tree_search(2,2) == expected)

    expected = ((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    assert(mnc_tree_search(3,3) == expected)   

    assert(mnc_tree_search(4, 4) == False)

#test_16()

# Task 1.7
def mnc_graph_search(m, c):
    '''
    Graph search requires to deal with the redundant path: cycle or loopy path.
    Modify the above implemented tree search algorithm to accelerate your AI.

    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    return mnc_tree_search(m, c)


# Test cases for Task 1.7
def test_17():
    expected = ((2, 0), (1, 0), (1, 1))
    assert(mnc_graph_search(2,1) == expected)

    expected = ((1, 1), (1, 0), (2, 0), (1, 0), (1, 1))
    assert(mnc_graph_search(2,2) == expected)

    expected = ((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    assert(mnc_graph_search(3,3) == expected)   

    assert(mnc_graph_search(4, 4) == False)

#test_17()
    

# Task 2.3
def pitcher_search(p1,p2,p3,a):
    '''
    Solution should be the action taken from the root node (initial state) to 
    the leaf node (goal state) in the search tree.

    Parameters
    ----------    
    p1: capacity of pitcher 1
    p2: capacity of pitcher 2
    p3: capacity of pitcher 3
    a: amount of water we want to measure
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a string: "Fill Pi", "Empty Pi", "Pi=>Pj". 
    If there is no solution, return False.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    raise NotImplementedError

# Test cases for Task 2.3
def test_23():
    expected = ('Fill P2', 'P2=>P1')
    assert(pitcher_search(2,3,4,1) == expected)

    expected = ('Fill P3', 'P3=>P1', 'Empty P1', 'P3=>P1')
    assert(pitcher_search(1,4,9,7) == expected)

    assert(pitcher_search(2,3,7,8) == False)

#test_23()

if __name__ == "__main__":
    print(mnc_tree_search(3,3))
    print(((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1)))
    # mnc_tree_search(3,3)
    