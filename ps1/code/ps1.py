from collections import deque
import copy
import time

class node:
    def __init__(self, arr, direction, action = -1, backtrack = None):
        self.arr = arr
        self.direction = direction # boolean True for left, False for right
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
    
    # BFS Approach

    # start = [[m,c],[0,0]]
    # goal = [[0,0],[m,c]]
    start_time = time.time()

    queue = deque()
    root = node([[m,c],[0,0]], True, -1, None)
    queue.append(root)
    while len(queue) != 0:
        curr = queue.popleft()

        # total possibilities of moving m's and c's from one side to another - 5
        for action in range(5):
            curr_array = copy.deepcopy(curr.arr)
            temp = node(curr_array, curr.direction, curr.action, curr.backtrack)
            next_state = transitionTo(temp, action, temp.direction)

            if not isValidState(next_state, m, c):
                continue # ignore and dont add to queue because state is invalid

            if next_state.arr == [[0,0],[m,c]]:
                solution = getSolution(next_state, [[m,c],[0,0]])
                solution = tuple(solution)
                print("My program took", time.time() - start_time, "to run")
                return solution
            queue.append(next_state)

    print("My program took", time.time() - start_time, "to run")
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
    newState = node(state.arr, state.direction, action, state.backtrack)
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

class nodeForGraph:
    def __init__(self, arr, action = -1, backtrack = None):
        self.arr = arr
        self.action = action
        self.backtrack = backtrack

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
    # BFS Approach

    # start = (m,c,0,0)
    # goal = (0,0,m,c)
    start_time = time.time()

    visited = set()
    queue = deque()
    root = nodeForGraph((m,c,0,0,True), -1, None)
    queue.append(root)
    while len(queue) != 0:
        curr = queue.popleft()


        # total possibilities of moving m's and c's from one side to another - 5
        for action in range(5):
            curr_array = copy.deepcopy(curr.arr)
            temp = nodeForGraph(curr_array, curr.action, curr)
            next_state = transitionToGraph(temp, action, temp.arr[4])

            # print("before")
            # print(next_state.arr)

            if not isValidStateGraph(next_state, m, c):
                continue # ignore and dont add to queue because state is invalid

            if next_state.arr in visited:
                continue

            # print("after")
            # print(next_state.arr)

            if next_state.arr == (0, 0, m, c, False):
                solution = getSolutionGraph(next_state, (m, c, 0, 0, True))
                solution = tuple(solution)
                print("My program took", time.time() - start_time, "to run")
                return solution
            queue.append(next_state)

        visited.add(curr.arr)

    print("My program took", time.time() - start_time, "to run")
    return False

# I define 5 possible choices that the boat can choose from when moving from one side to another
# 0: 1m
# 1: 1c
# 2: 1m1c
# 3: 2m
# 4: 2c

def getTransitionGraph(action):
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

def transitionToGraph(state, action, LR):
    lm = state.arr[0]
    lc = state.arr[1]
    rm = state.arr[2]
    rc = state.arr[3]

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
            lm = state.arr[0] + -1 * factor
            rm = state.arr[2] + 1 * factor
        case 1:
            # 1c
            lc = state.arr[1] + -1 * factor
            rc = state.arr[3] + 1 * factor
        case 2:
            # 1m1c
            lm = state.arr[0] + -1 * factor
            rm = state.arr[2] + 1 * factor
            lc = state.arr[1] + -1 * factor
            rc = state.arr[3] + 1 * factor
        case 3:
            # 2m
            lm = state.arr[0] + -2 * factor
            rm = state.arr[2] + 2 * factor
        case 4:
            #2c
            lc = state.arr[1] + -2 * factor
            rc = state.arr[3] + 2 * factor
    
    newState = nodeForGraph((lm, lc, rm, rc, not LR), action, state.backtrack)

    return newState

def isValidStateGraph(state, m, c):
    curr = state.arr

    # check if total number of cannibals and missionaries is less than m and c and more than 0
    if curr[0] < 0 or curr[0] > m or curr[2] < 0 or curr[2] > m:
        return False
    if curr[1] < 0 or curr[1] > c or curr[3] < 0 or curr[3] > c:
        return False

    # check if cannibals outnumber missionaries
    # side0
    if curr[0] < curr[1] and curr[0] != 0:
        return False

    if curr[2] < curr[3] and curr[2] != 0:
        return False
    
    return True

def getSolutionGraph(state, start):
    solution = []
    curr = state

    while (curr.arr != start):
        transitionTuple = getTransitionGraph(curr.action)
        
        solution.insert(0, transitionTuple)
        curr = curr.backtrack

        # if curr == None:
        #     solution.pop(0)
        #     return solution
    return solution

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
    print(mnc_graph_search(6,5))
    # print(((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1)))


"""
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
    LIMIT = 14
    depth = 0
    while depth < LIMIT:
        # print("iter")
        result = dfsIterative(m, c, depth)
        if result == False:
            depth += 1
        else:
            return result
    
    return False

def dfsIterative(m, c, limit):
    # DFS Iterative Approach

    # start = [[m,c],[0,0]]
    # goal = [[0,0],[m,c]]
    stack = deque()
    root = node([[m,c],[0,0]], True, False)
    stack.append(root)
    while len(stack) != 0:
        curr = stack.popleft()

        # total possibilities of moving m's and c's from one side to another - 5
        for action in range(5):
            curr_array = copy.deepcopy(curr.arr)
            temp = node(curr_array, curr.direction, curr.visited, curr.action, curr.backtrack)
            next_state = transitionTo(temp, action, temp.direction)

            if not isValidState(next_state, m, c):
                continue # ignore and dont add to stack because state is invalid

            if next_state.depth > limit:
                continue # ignore and dont add to stack because depth has exceeded

            if next_state.arr == [[0,0],[m,c]]:
                solution = getSolution(next_state, [[m,c],[0,0]])
                # solution = tuple(solution)
                return solution
            stack.append(next_state)
        
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
"""