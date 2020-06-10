import os
import functools
import math
import numpy as np
import random
import paddle
import paddle.fluid as fluid
from .img_tool import process_image

def image_dataset_from_filelist(filelist, configs, 
                                shuffle=True, phase="train"):
    
    loader = create_data_loader(phase, configs)
    reader = reader_creator(filelist, phase, configs, shuffle)
    batch_reader = paddle.batch(reader, configs.batch_size)
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    places = fluid.CUDAPlace(gpu_id) if configs.use_gpu else fluid.CPUPlace()
    loader.set_sample_list_generator(batch_reader, places)
    return loader


def reader_creator(filelist, phase, configs, shuffle, pass_id_as_seed=0):
    def _reader():
        data_dir = filelist[:-4]
        print(data_dir)
        with open(filelist) as flist:
            full_lines = [line.strip() for line in flist]
            if shuffle:
                if (not hasattr(_reader, 'seed')):
                    _reader.seed = pass_id_as_seed
                random.Random(_reader.seed).shuffle(full_lines)
                print("reader shuffle seed", _reader.seed)
                if _reader.seed is not None:
                    _reader.seed += 1

            if phase == 'train':
                trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
                if os.getenv("PADDLE_TRAINER_ENDPOINTS"):
                    trainer_count = len(os.getenv("PADDLE_TRAINER_ENDPOINTS").split(","))
                else:
                    trainer_count = int(os.getenv("PADDLE_TRAINERS", "1"))

                per_node_lines = int(math.ceil(len(full_lines) * 1.0 / trainer_count))
                total_lines = per_node_lines * trainer_count

                # aligned full_lines so that it can evenly divisible
                full_lines += full_lines[:(total_lines - len(full_lines))]
                assert len(full_lines) == total_lines

                # trainer get own sample
                lines = full_lines[trainer_id:total_lines:trainer_count]
                assert len(lines) == per_node_lines

                print("trainerid, trainer_count", trainer_id, trainer_count)
                print(
                    "read images from %d, length: %d, lines length: %d, total: %d"
                    % (trainer_id * per_node_lines, per_node_lines, len(lines),
                       len(full_lines)))
            else:
                print("mode is not train")
                lines = full_lines

            for line in lines:
                if phase == 'train':
                    img_path, label = line.split()
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(data_dir, img_path)
                    yield (img_path, int(label))
                elif phase == 'val':
                    img_path, label = line.split()
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(data_dir, img_path)
                    yield (img_path, int(label))
    image_mapper = functools.partial(
        process_image,
        settings=configs,
        mode=phase,
        color_jitter=False,
        rotate=False,
        crop_size=224, mean=configs.image_mean, std=configs.image_std)
    reader = paddle.reader.xmap_readers(
        image_mapper, _reader, 4, 4000, order=False)
    return reader


def create_data_loader(phase, configs, data_layout='NCHW'):

    image_shape = [int(m) for m in configs.image_shape.split(",")]
    if data_layout == "NHWC":
        image_shape=[image_shape[1], image_shape[2], image_shape[0]]
        feed_image = fluid.data(
            name="feed_image",
            shape=[None] + image_shape,
            dtype="float32",
            lod_level=0)
    else:
        # NCHW
        feed_image = fluid.data(
            name="feed_image",
            shape=[None] + image_shape,
            dtype="float32",
            lod_level=0)

    feed_label = fluid.data(
        name="feed_label", shape=[None, 1], dtype="int64", lod_level=0)
    feed_y_a = fluid.data(
        name="feed_y_a", shape=[None, 1], dtype="int64", lod_level=0)

    if phase == 'train' and configs.use_mixup:
        feed_y_b = fluid.data(
            name="feed_y_b", shape=[None, 1], dtype="int64", lod_level=0)
        feed_lam = fluid.data(
            name="feed_lam", shape=[None, 1], dtype="float32", lod_level=0)

        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[feed_image, feed_y_a, feed_y_b, feed_lam],
            capacity=64,
            use_double_buffer=True,
            iterable=True)
        return data_loader
    else:
        if configs.use_dali:
            return None

        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=[feed_image, feed_label],
            capacity=64,
            use_double_buffer=True,
            iterable=True)

        return data_loader


