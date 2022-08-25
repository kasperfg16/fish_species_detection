def distance_between_points(p1, p2):
    ''' Returns the distance between two points '''
    distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return abs(distance)