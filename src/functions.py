import numpy as np

def distanceCalc(p1, p2):
    distance = 0
    p1x = p1[0]
    p1y = p1[1]
    p2x = p2[0]
    p2y = p2[1]
    deltax = np.abs(p1x - p2x)
    deltay = np.abs(p1y - p2y)
    distance = np.sqrt(deltax ** 2 + deltay ** 2)

    return distance

def remove_outliers(lst, m=2):
    data = np.array(lst)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    data = data[abs(data - median) < m * mad]
    return np.ndarray.tolist(data)
