import numpy as np
import cv2
import random
import sys
import copy
import timeit
import math
from enum import Enum


def average(pixels):
    reds = [p[0] for p in pixels]
    greens = [p[1] for p in pixels]
    blues = [p[2] for p in pixels]

    r = sum(reds) // len(reds)
    g = sum(greens) // len(greens)
    b = sum(blues) // len(blues)
    return (r, g, b)


class Dist(Enum):
    MANHATTAN = 1
    EUCLIDEAN = 2
    CNC = 3

def distance(center, img_point, dist_type=Dist.MANHATTAN):
    if dist_type == Dist.CNC:
        L = abs(center[0] - img_point[0])
        a1 = center[1]
        a2 = img_point[1]
        b1 = center[2]
        b2 = img_point[2]
        C = math.sqrt(a1**2 + b1**2) - math.sqrt(a2**2 + b2**2)
        H = math.sqrt((a1 - a2)**2 + (b1 - b2)**2 + C**2)

        SL = .511 if (center[0] - img_point[0]) >=(16/100)*255 else (0.040975*(center[0] - img_point[0]) / (1 + .01765*(center[0] - img_point[0])))

        SC = .0638*(math.sqrt(a1**2 + b1**2))/ (1 + .0131*(math.sqrt(a1**2 + b1**2)) + .638)
        F = math.sqrt(math.sqrt(a1**2 + b1**2)**4 / (math.sqrt(a1**2 + b1**2)**4 + 1900))

        T = .56 + math.abs(.2 * math.cos(math.pi))
        SH = SC*(F*T + 1 - F)

        E = math.sqrt((L / SL)**2 + (H / SH)**2 + (C / SC)**2)

        return E

    if dist_type == Dist.MANHATTAN:
        r = abs(center[0] - img_point[0])
        g = abs(center[1] - img_point[1])
        b = abs(center[2] - img_point[2])
        return r + g + b

    if dist_type == Dist.EUCLIDEAN:
        r = abs(center[0] - img_point[0])
        g = abs(center[1] - img_point[1])
        b = abs(center[2] - img_point[2])

        return math.sqrt(r**2 + g**2 + b**2)


def k_means(img, k, iterations):
    img = img.tolist()

    centers = [(random.randint(0, len(img)), random.randint(0, len(img[0])))
               for i in range(0, k)]
    centers = [tuple(img[c[0]][c[1]]) for c in centers]

    for i in range(0, iterations):
        # print("iteration ", i)
        clusters = dict()
        for c in centers:
            clusters[c] = []

        for y in range(0, len(img)):
            for x in range(0, len(img[0])):
                shortest_dist = sys.maxsize
                closest_center = None
                for c in centers:
                    dist = distance(c, img[y][x], dist_type=Dist.EUCLIDEAN)
                    if dist < shortest_dist:
                        closest_center = c
                        shortest_dist = dist

                clusters[closest_center].append(img[y][x])

        for ci in range(0, len(centers)):
            a = average(clusters[centers[ci]])
            centers[ci] = a

    return centers


img = cv2.imread('china.png')
# Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(img)
Z = Z / 255
Z = cv2.cvtColor(Z, cv2.COLOR_BGR2LAB)
Z = Z.reshape((-1,3))

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, .0001)
K = 4

start = timeit.default_timer()
ret, label, center = cv2.kmeans(Z, K, None, criteria, 30,
                                cv2.KMEANS_RANDOM_CENTERS)

# center = np.uint8(center)
# center = k_means(img, 4, 30)

end = timeit.default_timer()

print(end - start)

# redraw the image with pallete you made:
res = center[label.flatten()]
res2 = res.reshape((img.shape))

# LAB white -> 255 128 128
display_img = cv2.copyMakeBorder(
    res2,
    0,
    100,
    0,
    0,
    borderType=cv2.BORDER_CONSTANT,
    value=(100,0,0,1))

for i, color in enumerate(center.tolist()):
# for i, color in enumerate(center):
    r, g, b = color
    color = (r, g, b)

    OFFSET_X = 20
    OFFSET_Y = 50
    start_x = i * OFFSET_X
    start_y = display_img.shape[0]
    cv2.rectangle(
        display_img, (start_x, start_y), (start_x + OFFSET_X - 1,
                                          start_y - OFFSET_Y),
        color,
        thickness=-1)

display_img = cv2.cvtColor(display_img, cv2.COLOR_LAB2BGR)
display_img = display_img * 255
display_img = np.uint8(display_img)

cv2.imshow('display_img', display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()