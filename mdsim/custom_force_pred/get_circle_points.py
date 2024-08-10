import math
pi = math.pi

# https://stackoverflow.com/questions/8487893/generate-all-the-points-on-the-circumference-of-a-circle
def PointsInCircum(r, n=100, x_center=0, y_center=0):
    return [(x_center+round(math.cos(2*pi/n*x)*r, 4), y_center+round(math.sin(2*pi/n*x)*r, 4)) for x in range(0,n)]

points = PointsInCircum(0.5, 500, 2, -0.5)

with open("mdsim/custom_force_pred/points_on_circle.txt", 'w') as outfile:
    for point in points:
        # print the values as space-separated x coordinate and y coordinate
        outfile.write(f'{"%.4f" % point[0]} {"%.4f" % point[1]}\n')