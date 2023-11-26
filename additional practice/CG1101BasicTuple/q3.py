ELEVATOR_SPEED = 2

def operate_elevator(t1, t2):
    # code to operate elevators
    dist1 = dist2 = 0
    final1 = final2 = 1

    if (t1[0] == 1):
        dist1 += getDistance(1, t1)
        final1 = t1[2]
    else:
        dist2 += getDistance(1, t1)
        final2 = t1[2]
    
    if (t2[0] == 1):
        dist1 += getDistance(final1, t2)
        final1 = t2[2]
    else:
        dist2 += getDistance(final2, t2)
        final2 = t2[2]

    time1 = dist1 * ELEVATOR_SPEED
    time2 = dist2 * ELEVATOR_SPEED

    res1 = (1, time1, final1)
    res2 = (2, time2, final2)
    return (res1, res2)

def getDistance(startfloor, t):
    return abs(startfloor - t[1]) + abs(t[1] - t[2])

if __name__ == "__main__":
    # ((1, 20, 7), (2, 14, 8))
    print(operate_elevator((2, 5, 8), (1, 9, 7)))