# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path
import copy
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import paddle
from ppfleetx.utils.log import logger
from ppfleetx.data.transforms.utils import create_preprocess_operators, transform

__all__ = [
    "GeneralClsDataset",
    "ImageFolder",
    "CIFAR10",
    "ContrativeLearningDataset",
]


class GeneralClsDataset(paddle.io.Dataset):
    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 delimiter=" ",
                 multi_label=False,
                 class_num=None):
        if multi_label:
            assert class_num is not None, "Must set class_num when multi_label=True"
        self.multi_label = multi_label
        self.classes_num = class_num

        self._img_root = image_root
        self._cls_path = cls_label_path
        self.delimiter = delimiter
        self._transform_ops = None
        if transform_ops:
            self._transform_ops = create_preprocess_operators(transform_ops)

        self.images = []
        self.labels = []
        self._load_anno()

    def _load_anno(self):
        assert os.path.exists(
            self._cls_path), f"{self._cls_path} does not exists"
        assert os.path.exists(
            self._img_root), f"{self._img_root} does not exists"
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for l in lines:
                l = l.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, l[0]))
                if self.multi_label:
                    self.labels.append(l[1])
                else:
                    self.labels.append(np.int32(l[1]))
                assert os.path.exists(self.images[
                    -1]), f"{self.images[-1]} does not exists"

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            if self.multi_label:
                one_hot = np.zeros([self.classes_num], dtype=np.float32)
                cls_idx = [int(e) for e in self.labels[idx].split(',')]
                for idx in cls_idx:
                    one_hot[idx] = 1.0
                return (img, one_hot)
            else:
                return (img, np.int32(self.labels[idx]))

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        if self.multi_label:
            return self.classes_num
        return len(set(self.labels))


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif",
                  ".tiff", ".webp")


class ImageFolder(paddle.io.Dataset):
    """ Code ref from https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py
    
    A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, extensions=IMG_EXTENSIONS, transform_ops=None):

        self.root = root
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions)
        logger.info(
            f'find total {len(classes)} classes and {len(samples)} images.')

        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = samples
        self.targets = [s[1] for s in samples]

        self._transform_ops = None
        if transform_ops:
            self._transform_ops = create_preprocess_operators(transform_ops)

    @staticmethod
    def make_dataset(
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None, ):
        """Generates a list of samples of a form (path_to_sample, class).

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")

        directory = os.path.expanduser(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time"
            )

        if extensions is not None:

            def is_valid_file(filename: str) -> bool:
                return filename.lower().endswith(
                    extensions
                    if isinstance(extensions, str) else tuple(extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(
                    os.walk(
                        target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def find_classes(self, directory):
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """

        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(
                f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, idx):
        try:
            path, target = self.imgs[idx]
            with open(path, 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)

            return (img, np.int32(target))

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(path, ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self) -> int:
        return len(self.imgs)

    @property
    def class_num(self):
        return len(set(self.classes))


class CIFAR10(paddle.io.Dataset):
    def __init__(
            self,
            root,
            mode='train',
            transform_ops=None, ):
        self.root = root
        self.mode = mode
        assert self.mode in ['train', 'test']
        self._transform_ops = None

        self.URL = 'https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz'

        if transform_ops:
            self._transform_ops = create_preprocess_operators(transform_ops)

        if not os.path.exists(os.path.join(self.root, f'data_batch_1')):
            from ppfleetx.utils.download import cached_path
            from ppfleetx.utils.file import untar
            zip_path = cached_path(
                self.URL, cache_dir=os.path.abspath(self.root))
            untar(
                zip_path,
                mode="r:gz",
                out_dir=os.path.join(self.root, '..'),
                delete=True)

        # wait to download dataset
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()

        self.images = []
        self.labels = []
        self._load_anno()

    def _load_anno(self):
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        if self.mode == 'train':
            for idx in range(1, 6):
                path = os.path.join(self.root, f'data_batch_{idx}')
                ret = unpickle(path)
                data = ret[b'data']
                labels = ret[b'labels']
                for i in range(len(data)):
                    img = data[i].reshape((3, 32, 32)).transpose((1, 2, 0))
                    self.images.append(img)
                    self.labels.append(labels[i])
        else:
            path = os.path.join(self.root, f'test_batch')
            ret = unpickle(path)
            data = ret[b'data']
            labels = ret[b'labels']
            for i in range(len(data)):
                img = data[i].reshape((3, 32, 32)).transpose((1, 2, 0))
                self.images.append(img)
                self.labels.append(labels[i])

    def __getitem__(self, idx):
        img = self.images[idx]
        if self._transform_ops:
            img = transform(img, self._transform_ops)

        return (img, np.int32(self.labels[idx]))

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(self.labels))


class ContrativeLearningDataset(ImageFolder):
    """ Code ref from https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py
    
    A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
    """

    def __init__(self, root, extensions=IMG_EXTENSIONS, transform_ops=None):
        super(ContrativeLearningDataset, self).__init__(
            root, extensions=extensions, transform_ops=transform_ops)

        # remove unused attr
        del self.classes
        del self.class_to_idx
        del self.targets
        # only use image path
        self.imgs = [s[0] for s in self.imgs]

    def __getitem__(self, idx):
        try:
            path = self.imgs[idx]
            with open(path, 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img1 = transform(img, self._transform_ops)
                img2 = transform(img, self._transform_ops)

            return img1, img2

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(path, ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self) -> int:
        return len(self.imgs)

    @property
    def class_num(self):
        raise NotImplementedError
