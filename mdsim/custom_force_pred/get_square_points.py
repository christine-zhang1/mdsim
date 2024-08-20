def points_on_square(r, n, x_center, y_center):
    # r is the length of half of each side. n is the number of points on each side
    all_points = [(x_center+d*r/n*2, y_center+r) for d in range(-n//2, n//2)]
    all_points.extend([(x_center+r, y_center-d*r/n*2) for d in range(-n//2, n//2)])
    all_points.extend([(x_center-d*r/n*2, y_center-r) for d in range(-n//2, n//2)])
    all_points.extend([(x_center-r, y_center+d*r/n*2) for d in range(-n//2, n//2)])
    return all_points

points = points_on_square(0.25, 200, 2, -0.5)

with open("mdsim/custom_force_pred/points_on_square.txt", 'w') as outfile:
    for point in points:
        # print the values as space-separated x coordinate and y coordinate
        outfile.write(f'{"%.4f" % point[0]} {"%.4f" % point[1]}\n')
