# odometer
# (final odometer val, (dist1,dist2,dist3,...))
def car(odometer, distances):

    odometerVal = odometer

    total_number_of_trips = 0
    total_distance = 0

    if len(distances) == 0:
        prevDist = 0
    else:
        prevDist = distances[0]

    max_diff_between_two_consecutive_trips = 0

    for dist in distances:
        max_diff_between_two_consecutive_trips = round(max(max_diff_between_two_consecutive_trips, abs(dist - prevDist)), 1)
        prevDist = dist
        total_distance += dist
        total_number_of_trips += 1

    if len(distances) < 2:
        max_diff_between_two_consecutive_trips = 0
    
    if total_number_of_trips != 0:
        avg_dist_per_trip = total_distance / total_number_of_trips
    else:
        avg_dist_per_trip = total_distance
    avg_dist_per_trip = round(avg_dist_per_trip, 1)
    
    final_odometer_value = cycDistance(odometerVal, total_distance)
    
    return (final_odometer_value, total_number_of_trips, avg_dist_per_trip, max_diff_between_two_consecutive_trips)

def cycDistance(initialVal, dist):
    if dist == 0:
        return round(initialVal, 1)
    return round((initialVal + dist) % 1000,1)
