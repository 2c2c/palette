import numpy as np
import cv2
import random
import sys
import os
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

        SL = .511 if (center[0] - img_point[0]) >= (16 / 100) * 255 else (
            0.040975 * (center[0] - img_point[0]) /
            (1 + .01765 * (center[0] - img_point[0])))

        SC = .0638 * (math.sqrt(a1**2 + b1**2)) / (
            1 + .0131 * (math.sqrt(a1**2 + b1**2)) + .638)
        F = math.sqrt(
            math.sqrt(a1**2 + b1**2)**4 / (math.sqrt(a1**2 + b1**2)**4 + 1900))

        T = .56 + math.abs(.2 * math.cos(math.pi))
        SH = SC * (F * T + 1 - F)

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


def make_palette(img, K):
    # we convert from rgb -> lab colorspace. it performs better in quantification
    # need to map rgb 0-255 to 0-1 before converting
    # to display image do the reverse
    Z = np.float32(img)
    Z = Z / 255
    Z = cv2.cvtColor(Z, cv2.COLOR_BGR2LAB)
    #flatten for kmeans
    Z = Z.reshape((-1, 3))
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, .0001)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 30,
                                    cv2.KMEANS_RANDOM_CENTERS)

    return label, center


def quantize(img, label, center):
    # redraw the image with palette you made:
    res = center[label.flatten()]
    display_img = res.reshape((img.shape))

    display_img = cv2.cvtColor(display_img, cv2.COLOR_LAB2BGR)
    display_img = display_img * 255
    display_img = np.uint8(display_img)

    return display_img


def lab2bgr(colors):
    """ hacky method of using opencv to take a list of colors from lab to bgr"""

    colors = np.float32([np.float32([c, c, c]) for c in colors])
    colors = cv2.cvtColor(colors, cv2.COLOR_LAB2BGR)
    colors = colors * 255
    colors = np.uint8(colors)
    colors = colors.reshape(-1, 3)
    colors = np.unique(colors, axis=0)

    return colors


def append_palette(img, center):
    # LAB white -> 255 128 128 on uint8
    # LAB white -> 100 0 0 on float32
    display_img = cv2.copyMakeBorder(
        img,
        100,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255, 1))

    center = lab2bgr(center)

    for i, color in enumerate(center.tolist()):
        # for i, color in enumerate(center):
        b, g, r = color
        color = (b, g, r)

        max_width = display_img.shape[1]
        OFFSET_X = max_width // len(center)
        OFFSET_Y = 50

        start_x = OFFSET_X * i + 1
        start_y = 25
        cv2.rectangle(
            display_img, (start_x, start_y), (start_x + OFFSET_X - 1,
                                              start_y + OFFSET_Y),
            color,
            thickness=-1)

    return display_img

def collage(filename, start, stop):
    img = cv2.imread(filename)
    num_quants = stop - start + 1 
    q_imgs = []
    for i in range(start, stop + 1):
        l, c = make_palette(img, i)
        q = quantize(img, l, c)
        q = cv2.resize(
            q,
            None,
            fx=1 / num_quants,
            fy=1 / num_quants,
            interpolation=cv2.INTER_CUBIC)
        q = append_palette(q, c)
        q_imgs.append(q)

    q_imgs = tuple(q_imgs)

    merged = np.concatenate(q_imgs, axis=1)
    merged = cv2.resize(
        merged, (img.shape[1], merged.shape[0]), interpolation=cv2.INTER_CUBIC)
    merged = np.concatenate((img, merged), axis=0)

    name, extension = filename.split(".")
    name = name.split("\\")[-1]

    cv2.imwrite(f"output/{name}_palette.{extension}", q)


collage(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))