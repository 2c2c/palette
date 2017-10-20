import numpy as np
import cv2
import random
import sys
import copy
import timeit


def average(pixels):
    reds = [p[0] for p in pixels]
    greens = [p[1] for p in pixels]
    blues = [p[2] for p in pixels]

    r = sum(reds) // len(reds)
    g = sum(greens) // len(greens)
    b = sum(blues) // len(blues)
    return (r, g, b)


def distance(center, img_point):
    r = abs(center[0] - img_point[0])
    g = abs(center[1] - img_point[1])
    b = abs(center[2] - img_point[2])
    return r + g + b


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
                    dist = distance(c, img[y][x])
                    if dist < shortest_dist:
                        closest_center = c
                        shortest_dist = dist

                clusters[closest_center].append(img[y][x])

        for ci in range(0, len(centers)):
            a = average(clusters[centers[ci]])
            centers[ci] = a

    return centers

img = cv2.imread('papika.jpg')
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5

start = timeit.default_timer()
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10,
                                cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center).tolist()
# center = k_means(img, 5, 10)

end = timeit.default_timer()

print(end - start)

# redraw the image with pallete you made:
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))

display_img = cv2.copyMakeBorder(
    img, 0, 100, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255, 1))

# for i, color in enumerate(center.tolist()):
for i, color in enumerate(center):
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
cv2.imshow('display_img', display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()