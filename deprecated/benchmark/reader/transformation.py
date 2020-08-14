#!/usr/bin/env python
#coding:utf-8
from __future__ import division
from __future__ import print_function
import numpy as np
import math
from PIL import Image
import cv2


def resize_short(img, target_size):
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resize_width = int(round(img.shape[1] * percent))
    resize_height = int(round(img.shape[0] * percent))
    img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    return img


def crop_image(img, target_size, center=True):
    width, height = img.shape[1], img.shape[0]
    size = target_size
    if center:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end]
    return img


def random_crop(img, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
    """ random_crop """
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.shape[0]) / img.shape[1]) / (w**2), (float(img.shape[1]) / img.shape[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.shape[0] * img.shape[1] * np.random.uniform(scale_min, scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)
    i = np.random.randint(0, img.shape[0] - w + 1)
    j = np.random.randint(0, img.shape[1] - h + 1)

    img = img[i:i + w, j:j + h, :]

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return img


def rotate_image(img):
    """ rotate_image """
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    angle = np.random.randint(-10, 11)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h))
    return rotated


def distort_color(img, color_pca=True):
    def random_brightness(image, lower=0.5, upper=1.5):
        image = np.clip(image, 0.0, 1.0)
        e = np.random.uniform(lower, upper)
        # zero = np.zeros([1] * len(img.shape), dtype=img.dtype)
        return image * e  # + zero * (1.0 - e)

    def random_contrast(image, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        gray = np.mean(image[0]) * 0.299 + np.mean(image[1]) * 0.587 + np.mean(image[2]) * 0.114
        mean = np.ones([1] * len(image.shape), dtype=image.dtype) * gray
        return image * e + mean * (1.0 - e)

    def random_color(image, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        gray = image[0] * 0.299 + image[1] * 0.587 + image[2] * 0.114
        gray = np.expand_dims(gray, axis=0)
        return image * e + gray * (1.0 - e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    if color_pca:
        eigvec = np.array([ [-0.5675,  0.7192,  0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948,  0.4203] ], dtype='float32')
        alpha = (np.random.rand(3) * 0.1).astype('float32')
        eigval = np.array([0.2175, 0.0188, 0.0045], dtype='float32')
        rgb = np.sum(eigvec * alpha * eigval, axis=1)
        img += rgb.reshape([3, 1, 1])

    return img


img_mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape((3, 1, 1))


def std_image(img):
    img -= img_mean
    img *= (1.0 / img_std)
    return img


