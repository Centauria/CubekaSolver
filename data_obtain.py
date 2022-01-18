# -*- coding: utf-8 -*-
import argparse
import os
import string
from itertools import permutations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from tqdm import tqdm

import alphabet


def load_cubeka(path):
    files = os.listdir(path)
    data = dict()
    for f in files:
        x = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
        _, x = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY_INV)
        data[os.path.splitext(f)[0]] = x
    return data


def show(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def diff(image_a, image_b):
    kernel = np.ones((3, 3), np.uint8)
    x = cv2.bitwise_and(image_a, image_b)
    y = cv2.bitwise_or(image_a, image_b)
    x = cv2.bitwise_xor(x, y)
    # x = cv2.morphologyEx(z, cv2.MORPH_OPEN, kernel)
    return x


def markers(image_a, image_b):
    equal_region = cv2.bitwise_and(image_a, image_b)
    diff_region = diff(image_a, image_b)
    markers = np.zeros_like(image_a, dtype=int)
    markers[np.where(equal_region > 127)] = -1
    markers[np.where(diff_region > 127)] = 1
    return markers


def margin(matrix, smooth=105):
    x, y = np.nonzero(matrix)
    lr = []
    for i in range(matrix.shape[0]):
        w = y[np.where(x == i)]
        lr.append([matrix[i, np.min(w)], matrix[i, np.max(w)]])
    lr = np.array(lr)
    left, right = lr[:, 0], lr[:, 1]
    left = left[::-1]
    tb = []
    for j in range(matrix.shape[1]):
        w = x[np.where(y == j)]
        tb.append([matrix[np.min(w), j], matrix[np.max(w), j]])
    tb = np.array(tb)
    top, bottom = tb[:, 0], tb[:, 1]
    left, top, right, bottom = list(map(lambda x: medfilt(255 * np.maximum(x, 0), smooth), [left, top, right, bottom]))
    return left, top, right, bottom


def show_diff(image_a, image_b):
    x_margin = margin(markers(image_a, image_b))
    plt.rcParams['figure.figsize'] = (20, 5)
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        w = np.where(x_margin[i] > 127)
        plt.scatter(w, 1 * np.ones_like(w))
        plt.grid(axis='x')
        plt.xlim(0, x_margin[i].shape[0])
    plt.show()


def encode(margin_array, dpi=100):
    x = margin_array.reshape((-1, dpi // 2))
    x = np.mean(x, axis=1)
    x[x < 127] = 0
    x[x > 127] = 1
    return x.astype(int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cubekasolver')
    parser.add_argument('output')
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--save-fig', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    alpha = alphabet.Alphabet('constructor.yaml', dpi=args.dpi, thickness=1.0)
    bar = tqdm(permutations(string.ascii_uppercase, 2))
    diff_matrix = [
        np.zeros((26, 26, 22)),
        np.zeros((26, 26, 12)),
        np.zeros((26, 26, 22)),
        np.zeros((26, 26, 12))
    ]
    for a, b in bar:
        margins = margin(markers(alpha.render(a), alpha.render(b)))
        encoded_margins = [encode(m, args.dpi) for m in margins]
        for i in range(len(margins)):
            diff_matrix[i][ord(a) - ord('A'), ord(b) - ord('A'), :] = encoded_margins[i]
        if args.save_fig:
            os.makedirs(os.path.join(args.output, 'figures'), exist_ok=True)
            fig = plt.figure(figsize=(20, 3))
            for i in range(len(margins)):
                w = np.where(margins[i] > 127)
                plt.subplot(1, 4, i + 1)
                plt.scatter(w, 1 * np.ones_like(w))
                plt.grid(axis='x')
                plt.xlim(0, margins[i].shape[0])
            fig.savefig(os.path.join(args.output, 'figures', f'{a}{b}.png'))
            plt.close(fig)
        bar.set_description(f'{a}{b}')
    for i in range(len(diff_matrix)):
        np.save(os.path.join(args.output, f'diff_matrix_{i}'), diff_matrix[i])
