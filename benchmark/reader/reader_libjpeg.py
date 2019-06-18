"""
Reader using libjpeg_turbo.
"""
from __future__ import division
from __future__ import print_function
import os
import random
import functools
import math
import numpy as np
import paddle
from turbojpeg import TurboJPEG, TJPF_RGB
from transformation import rotate_image, random_crop, distort_color
from transformation import resize_short, crop_image

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 102400

DATA_DIR = 'data/ILSVRC2012'

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

turbojpeg = TurboJPEG('./lib/libturbojpeg.so.0.2.0')


def process_image(sample, mode, color_jitter=False, rotate=True, mean=img_mean, std=img_std):
    img_path = sample[0]

    # don't support the jpeg image with a CMYK color space
    file = open(img_path, 'rb')
    img = turbojpeg.decode(file.read(), pixel_format=TJPF_RGB)
    file.close()
    if not (img is not None and len(img.shape) == 3 and img.shape[-1] == 3
            and img.max() > 0 and img.shape[0] > 5 and img.shape[1] > 5):
        print("image decode error! (from turbojpeg)")
        return None

    if mode == 'train':
        if rotate: 
            img = rotate_image(img)
        img = random_crop(img, DATA_DIM)
        if color_jitter:
            img = distort_color(img)
        if np.random.randint(0, 2) == 1:
            img = img[:, ::-1, :]
    else:
        img = resize_short(img, target_size=256)
        img = crop_image(img, target_size=DATA_DIM, center=True)
        #img = img.astype('float32').transpose((2, 0, 1)) * (1.0 / 255)

    img = img.astype('float32').transpose((2, 0, 1)) / 255
    img -= mean
    img /= std

    if mode == 'train' or mode == 'val':
        return img, sample[1]
    elif mode == 'test':
        return [img]
    else:
        raise ValueError("Unknown mode {}".format(mode))


def _reader_creator(file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=DATA_DIR,
                    pass_id_as_seed=1,
                    num_threads=THREAD):
    def reader():
        with open(file_list) as f_list:
            full_lines = [line.strip() for line in f_list]
            if shuffle:
                np.random.seed(pass_id_as_seed)
                np.random.shuffle(full_lines)

            for line in full_lines:
                if mode == 'train' or mode == 'val':
                    img_path, label = line.split()
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(data_dir, 'train', img_path)
                    yield img_path, int(label)
                elif mode == 'test':
                    img_path, label = line.split()
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(data_dir, 'val', img_path)
                    yield [img_path]

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, num_threads, BUF_SIZE, order=False)


def train(data_dir=DATA_DIR, pass_id_as_seed=1, num_threads=THREAD):
    file_list = os.path.join(data_dir, 'train.txt')
    return _reader_creator(
        file_list,
        'train',
        shuffle=True,
        color_jitter=False,
        rotate=False,
        data_dir=data_dir,
        pass_id_as_seed=pass_id_as_seed,
        num_threads=num_threads)
