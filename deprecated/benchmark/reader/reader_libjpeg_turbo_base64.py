"""
Reader using base64 with libjpeg_turbo.
"""
from __future__ import division
from __future__ import print_function
import os
import pybase64 as base64
import numpy as np
from turbojpeg import TurboJPEG, TJPF_RGB
import cv2
from transformation import rotate_image, random_crop, distort_color
from transformation import resize_short, crop_image
import paddle
import functools

turbojpeg = TurboJPEG('./lib/libturbojpeg.so.0.2.0')
TOKEN_IDX = [0, 1]
THREAD = 8
DATA_DIM = 224
BUF_SIZE = 102400
img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

DATA_DIR = 'data/ILSVRC2012'


def _process_image(img, mode, color_jitter=False, rotate=True, data_dim=DATA_DIM, mean=img_mean, std=img_std):
    try:
        # don't support the jpeg image with a CMYK color space
        img = turbojpeg.decode(img, pixel_format=TJPF_RGB)
        if not (img is not None and len(img.shape) == 3 and img.shape[-1] == 3
                and img.max() > 0 and img.shape[0] > 5 and img.shape[1] > 5):
            print("image decode error! (from turbojpeg)")
            return None
    except:
        img = cv2.imdecode(np.asarray(bytearray(img), dtype='uint8'), cv2.IMREAD_COLOR)
        if not (img is not None and len(img.shape) == 3 and img.shape[-1] == 3
                and img.max() > 0 and img.shape[0] > 5 and img.shape[1] > 5):
            print("image decode error! (from cv2)")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BRG2RGB)

    if mode == 'train':
        if rotate:
            img = rotate_image(img)
        img = random_crop(img, data_dim)
        img = img.astype('float32').transpose((2, 0, 1)) * (1.0 / 255)
        if color_jitter:
            img = distort_color(img)
        if np.random.randint(0, 2) == 1:
            img = img[:, ::-1, :]
    else:
        img = resize_short(img, target_size=256)
        img = crop_image(img, target_size=data_dim, center=True)
        img = img.astype('float32').transpose((2, 0, 1)) * (1.0 / 255)

    img -= mean
    img /= std
    return img


def _process_line(line, token_idx=TOKEN_IDX, mode="train", color_jitter=False, rotate=False, data_dim=DATA_DIM):
    tokens_data = line.strip().split('\t')
    img = base64.b64decode(tokens_data[token_idx[0]].replace('-', '+').replace('_', '/'))
    img = _process_image(img, mode, color_jitter, rotate, data_dim)
    if img is None:
        return []
    label = int(tokens_data[token_idx[1]])
    sample = [img, label]
    return sample


def _reader_creator(file_name,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=DATA_DIR,
                    pass_id_as_seed=1,
                    num_threads=THREAD):
    def reader():
        filename = os.path.join(data_dir, file_name)
        with open(filename) as f_list:
            full_lines = [line.strip() for line in f_list]
            if shuffle:
                np.random.seed(pass_id_as_seed)
                np.random.shuffle(full_lines)

            for line in full_lines:
                yield line

    mapper = functools.partial(
        _process_line, token_idx=TOKEN_IDX, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, num_threads, BUF_SIZE, order=False)


def train(data_dir=DATA_DIR, file_name='base64', pass_id_as_seed=1, num_threads=THREAD):
    return _reader_creator(
        file_name,
        'train',
        shuffle=True,
        color_jitter=False,
        rotate=True,
        data_dir=data_dir,
        pass_id_as_seed=pass_id_as_seed,
        num_threads=num_threads)
