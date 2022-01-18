# -*- coding: utf-8 -*-
import cv2.flann
import numpy as np
import yaml


class Alphabet:
    def __init__(self, constructor_file, ratio=(11, 6), dpi=100, thickness=1.0):
        with open(constructor_file) as f:
            self.constructor = yaml.load(f, Loader=yaml.CLoader)
        self.ratio = ratio
        self.dpi = dpi
        self.thickness = thickness
        self.shape = (ratio[0] * dpi, ratio[1] * dpi)

    def render(self, key):
        canvas = np.zeros(self.shape)
        if 'lines' in self.constructor[key].keys():
            lines = self.constructor[key]['lines']
        else:
            lines = []
        if 'arcs' in self.constructor[key].keys():
            arcs = self.constructor[key]['arcs']
        else:
            arcs = []
        for pt1, pt2 in lines:
            pt1 = np.array(pt1).flatten() * self.dpi
            pt2 = np.array(pt2).flatten() * self.dpi
            cv2.line(canvas, pt1.astype(int), pt2.astype(int), color=255, thickness=int(self.dpi * self.thickness))
        for center, axes, angle, start_angle, end_angle in arcs:
            center = np.array(center).flatten() * self.dpi
            axes = np.array(axes).flatten() * self.dpi
            cv2.ellipse(canvas, center.astype(int), axes.astype(int), angle, start_angle, end_angle, color=255,
                        thickness=int(self.dpi * self.thickness))
        return canvas
