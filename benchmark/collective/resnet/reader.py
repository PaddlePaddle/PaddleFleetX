from __future__ import division
import os
import math
import random
import functools
import cv2
import numpy as np
import paddle

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 1024

DATA_DIR = 'data/ILSVRC2012'

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    img = cv2.resize(img, (resized_width, resized_height))
    return img


def crop_image(img, target_size, center):
    height, width = img.shape[:2]
    size = target_size
    if center:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.shape[0]) / img.shape[1]) / (h**2),
                (float(img.shape[1]) / img.shape[0]) / (w**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.shape[0] * img.shape[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.shape[0] - h + 1)
    j = np.random.randint(0, img.shape[1] - w + 1)

    img = img[i:i+h, j:j+w, :]
    img = cv2.resize(img, (size, size))
    return img


def rotate_image(img):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle = np.random.randint(-10, 11)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def distort_color(img):
    return img


def process_image(sample, mode, color_jitter, rotate):
    img_path = sample[0]
    # img_data = np.frombuffer(img_path, dtype='uint8')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

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

    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255

    img -= img_mean
    img /= img_std

    if mode == 'train' or mode == 'val':
        return img, sample[1]
    elif mode == 'test':
        return [img]


def _reader_creator(file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=DATA_DIR,
                    pass_id_as_seed=0,
                    infinite=False,
                    num_trainers=1,
                    trainer_id=0):
    def reader():
        with open(file_list) as flist:
            full_lines = [line.strip() for line in flist]
            pass_id_as_seed_counter = pass_id_as_seed
            while True:
                if shuffle:
                    if pass_id_as_seed_counter:
                        np.random.seed(pass_id_as_seed_counter)
                    np.random.shuffle(full_lines)
                if (mode == 'train' or mode == 'parallel_val') and num_trainers > 1:
                    # distributed mode if more than one trainers
                    lines_per_trainer = len(full_lines) // num_trainers
                    lines = full_lines[trainer_id * lines_per_trainer:(trainer_id + 1)
                                    * lines_per_trainer]
                    print(
                        "read images from %d, length: %d, lines length: %d, total: %d"
                        % (trainer_id * lines_per_trainer, lines_per_trainer, len(lines),
                        len(full_lines)))
                else:
                    lines = full_lines

                for line in lines:
                    if mode == 'train' or mode == 'val' or mode == 'parallel_val':
                        subdir = mode
                        if mode == "parallel_val":
                            subdir = 'val'
                        img_path, label = line.split()
                        img_path = img_path.replace("JPEG", 'jpeg')
                        img_path = os.path.join(data_dir, subdir, img_path)
                        yield img_path, int(label)
                    elif mode == 'test':
                        img_path, label = line.split()
                        img_path = img_path.replace("JPEG", 'jpeg')
                        img_path = os.path.join(data_dir, 'val', img_path)
                        yield [img_path]
                if not infinite:
                    break
                pass_id_as_seed_counter += 1
                print("passid ++, current: ", pass_id_as_seed_counter)

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def train(data_dir=DATA_DIR, pass_id_as_seed=0, infinite=False, num_trainers=1, trainer_id=0):
    file_list = os.path.join(data_dir, 'train.txt')
    return _reader_creator(
        file_list,
        'train',
        shuffle=True,
        color_jitter=False,
        rotate=False,
        data_dir=data_dir,
        pass_id_as_seed=pass_id_as_seed,
        infinite=infinite,
        num_trainers=num_trainers,
        trainer_id=trainer_id)


def val(data_dir=DATA_DIR, num_trainers=1, trainer_id=0, parallel_test=False):
    file_list = os.path.join(data_dir, 'val.txt')
    mode = 'parallel_val' if parallel_test else 'val'
    return _reader_creator(file_list, mode, shuffle=False, 
            data_dir=data_dir, num_trainers=num_trainers, trainer_id=trainer_id)


def test(data_dir=DATA_DIR, num_trainers=1, trainer_id=0, parallel_test=False):
    file_list = os.path.join(data_dir, 'val.txt')
    return _reader_creator(file_list, 'test', shuffle=False, 
            data_dir=data_dir, num_trainers=num_trainers, trainer_id=trainer_id)
