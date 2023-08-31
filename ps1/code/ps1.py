from collections import deque
import copy

class node:
    def __init__(self, arr, direction, visited, action = -1, backtrack = None, depth = 0):
        self.arr = arr
        self.direction = direction # boolean True for left, False for right
        self.visited = visited # boolean
        self.action = action
        self.backtrack = backtrack
        if backtrack != None:
            self.depth = self.backtrack.depth + 1
        else:
            self.depth = depth

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
    

class pitcherNode:
    def __init__(self, capacityTuple, currentTuple, backtrack = None, action = None):
        self.capacityTuple = capacityTuple
        self.currentTuple = currentTuple
        self.backtrack = backtrack
        self.action = action

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

    # Memoized BFS Approach
    visited = set()
    queue = deque()
    root = pitcherNode((p1,p2,p3), (0,0,0))
    queue.append(root)
    while len(queue) != 0:
        curr = queue.popleft()

        print(len(queue))

        allChildren = possibleActions(curr)

        for child in allChildren:
            if not isValid(child):
                continue # ignore and dont add to queue because state is invalid

            if child.currentTuple in visited:
                continue

            if isGoal(child, a):
                sol = tuple(getSolution(child, a, (0, 0, 0)))
                # print(sol)
                return sol
            
            queue.append(child)

        visited.add((curr.currentTuple[0],
                    curr.currentTuple[1],
                    curr.currentTuple[2]))    

    return False

def flatten(l):
    return [item for sublist in l for item in sublist]

def possibleActions(curr):
    allPossibleActions = []

    allPossibleActions.append(filljugs(curr))
    allPossibleActions.append(emptyjugs(curr))
    allPossibleActions.append(transferwater(curr))

    allPossibleActions = flatten(allPossibleActions)

    return allPossibleActions


def isValid(curr):
    if 0 <= curr.currentTuple[0] <= curr.capacityTuple[0]:
        if 0 <= curr.currentTuple[1] <= curr.capacityTuple[1]:
            if 0 <= curr.currentTuple[2] <= curr.capacityTuple[2]:
                return True
    return False

def filljugs(curr):
    return [pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.capacityTuple[0],
                        curr.currentTuple[1],
                        curr.currentTuple[2]), curr, (0,0,0)),
                        
            pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0],
                        curr.capacityTuple[1],
                        curr.currentTuple[2]), curr, (0,1,0)),

            pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[1],
                        curr.currentTuple[1],
                        curr.capacityTuple[2]), curr, (0,2,0))
            ]

def emptyjugs(curr):
    return [pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (0,
                        curr.currentTuple[1],
                        curr.currentTuple[2]), curr, (1,0,0)),
                        
            pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0],
                        0,
                        curr.currentTuple[2]), curr, (1,1,0)),

            pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0],
                        curr.currentTuple[1],
                        0), curr, (1,2,0))
            ]

def transferwater(curr):
    final = []
    
    # a transfer b
    missing = curr.capacityTuple[1] - curr.currentTuple[1]
    if missing > curr.capacityTuple[0]:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (0,
                        curr.currentTuple[0] + curr.currentTuple[1],
                        curr.currentTuple[2]), curr, (2,0,0)))
    else:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0] - missing,
                        curr.capacityTuple[1],
                        curr.currentTuple[2]), curr, (2,0,0)))
    
    # b transfer c
    missing = curr.capacityTuple[2] - curr.currentTuple[2]
    if missing > curr.capacityTuple[1]:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0],
                        0,
                        curr.currentTuple[1] + curr.currentTuple[2]), curr, (2,0,1)))
    else:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0] ,
                        curr.currentTuple[1] - missing,
                        curr.capacityTuple[2]), curr, (2,0,1)))

    # c transfer a
    missing = curr.capacityTuple[0] - curr.currentTuple[0]
    if missing > curr.capacityTuple[2]:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0] + currentTuple[2],
                        curr.currentTuple[1],
                        0), curr, (2,0,2)))
    else:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.capacityTuple[0],
                        curr.currentTuple[1],
                        curr.currentTuple[2] - missing), curr, (2,0,2)))

    # a transfer c
    missing = curr.capacityTuple[2] - curr.currentTuple[2]
    if missing > curr.capacityTuple[0]:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (0,
                        curr.currentTuple[1],
                        curr.currentTuple[0] + curr.currentTuple[2]), curr, (2,0,3)))
    else:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0] - missing,
                        curr.currentTuple[1],
                        curr.capacityTuple[2]), curr, (2,0,3)))

    # c transfer b
    missing = curr.capacityTuple[1] - curr.currentTuple[1]
    if missing > curr.capacityTuple[2]:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0],
                        curr.currentTuple[1] + currentTuple[2],
                        0), curr, (2,0,4)))
    else:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0],
                        curr.capacityTuple[1],
                        curr.currentTuple[2] - missing), curr, (2,0,4)))

    # b transfer a
    missing = curr.capacityTuple[0] - curr.currentTuple[0]
    if missing > curr.capacityTuple[1]:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.currentTuple[0] + curr.currentTuple[1],
                        0,
                        curr.currentTuple[2]), curr, (2,0,5)))
    else:
        final.append(pitcherNode((curr.capacityTuple[0],
                        curr.capacityTuple[1],
                        curr.capacityTuple[2]),
                        (curr.capacityTuple[0],
                        curr.currentTuple[1] - missing,
                        curr.currentTuple[2]), curr, (2,0,5)))

    return final

def isGoal(curr, a):
    if curr.currentTuple[0] == a or curr.currentTuple[1] == a or curr.currentTuple[2] == a:
        return True
    return False

# 0 - Move (Fill, Empty, Transfer)
# 1 - Jugs (1,2,3)
# 2 - Direction (1-2, 2-3, 3-1, 1-3, 3-2, 2-1)

def getAction(curr):
    string = ""
    move = curr.action[0]
    jug = curr.action[1]
    direction = curr.action[2]

    moves = ("Fill", "Empty")
    jugs = ("P1", "P2", "P3")
    directions = ("P1=>P2", "P2=>P3", "P3=>P1", "P1=>P3", "P3=>P2", "P2=>P1")

    if move == 0:
        string = moves[0] + " " + jugs[jug]
    elif move == 1:
        string = moves[1] + " " + jugs[jug]
    else:
        string = directions[direction]

    return string

def getSolution(curr, a, initial):
    solution = []
    while(curr.currentTuple != initial):
        solution.insert(0, getAction(curr))
        curr = curr.backtrack
    return solution

# Test cases for Task 2.3
def test_23():
    expected = ('Fill P2', 'P2=>P1')
    assert(pitcher_search(2,3,4,1) == expected)

    expected = ('Fill P3', 'P3=>P1', 'Empty P1', 'P3=>P1')
    assert(pitcher_search(1,4,9,7) == expected)

    assert(pitcher_search(2,3,7,8) == False)

#test_23()

if __name__ == "__main__":
    pitcher_search(2,3,7,8)
    



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