# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys
from PIL import Image

import paddle
from paddle.io import Dataset
from paddle.utils import try_import

__all__ = []


def has_valid_extension(filename, extensions):
    """Checks if a file is a vilid extension.

    Args:
        filename (str): path to a file
        extensions (list[str]|tuple[str]): extensions to consider

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    assert isinstance(extensions,
                      (list, tuple)), ("`extensions` must be list or tuple.")
    extensions = tuple([x.lower() for x in extensions])
    return filename.lower().endswith(extensions)


def make_dataset(dir, class_to_idx, extensions, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)

    if extensions is not None:

        def is_valid_file(x):
            return has_valid_extension(x, extensions)

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(Dataset):
    """A generic data loader where the samples are arranged in this way:

    .. code-block:: text

        root/class_a/1.ext
        root/class_a/2.ext
        root/class_a/3.ext

        root/class_b/123.ext
        root/class_b/456.ext
        root/class_b/789.ext

    Args:
        root (str): Root directory path.
        loader (Callable, optional): A function to load a sample given its path. Default: None.
        extensions (list[str]|tuple[str], optional): A list of allowed extensions.
            Both :attr:`extensions` and :attr:`is_valid_file` should not be passed.
            If this value is not set, the default is to use ('.jpg', '.jpeg', '.png',
            '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'). Default: None.
        transform (Callable, optional): A function/transform that takes in
            a sample and returns a transformed version. Default: None.
        is_valid_file (Callable, optional): A function that takes path of a file
            and check if the file is a valid file. Both :attr:`extensions` and
            :attr:`is_valid_file` should not be passed. Default: None.

    Returns:
        :ref:`api_paddle_io_Dataset`. An instance of DatasetFolder.

    Attributes:
        classes (list[str]): List of the class names.
        class_to_idx (dict[str, int]): Dict with items (class_name, class_index).
        samples (list[tuple[str, int]]): List of (sample_path, class_index) tuples.
        targets (list[int]): The class_index value for each image in the dataset.

    Example:

        .. code-block:: python

            import shutil
            import tempfile
            import cv2
            import numpy as np
            import paddle.vision.transforms as T
            from pathlib import Path
            from paddle.vision.datasets import DatasetFolder


            def make_fake_file(img_path: str):
                if img_path.endswith((".jpg", ".png", ".jpeg")):
                    fake_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                    cv2.imwrite(img_path, fake_img)
                elif img_path.endswith(".txt"):
                    with open(img_path, "w") as f:
                        f.write("This is a fake file.")

            def make_directory(root, directory_hierarchy, file_maker=make_fake_file):
                root = Path(root)
                root.mkdir(parents=True, exist_ok=True)
                for subpath in directory_hierarchy:
                    if isinstance(subpath, str):
                        filepath = root / subpath
                        file_maker(str(filepath))
                    else:
                        dirname = list(subpath.keys())[0]
                        make_directory(root / dirname, subpath[dirname])

            directory_hirerarchy = [
                {"class_0": [
                    "abc.jpg",
                    "def.png"]},
                {"class_1": [
                    "ghi.jpeg",
                    "jkl.png",
                    {"mno": [
                        "pqr.jpeg",
                        "stu.jpg"]}]},
                "this_will_be_ignored.txt",
            ]

            # You can replace this with any directory to explore the structure
            # of generated data. e.g. fake_data_dir = "./temp_dir"
            fake_data_dir = tempfile.mkdtemp()
            make_directory(fake_data_dir, directory_hirerarchy)
            data_folder_1 = DatasetFolder(fake_data_dir)
            print(data_folder_1.classes)
            # ['class_0', 'class_1']
            print(data_folder_1.class_to_idx)
            # {'class_0': 0, 'class_1': 1}
            print(data_folder_1.samples)
            # [('./temp_dir/class_0/abc.jpg', 0), ('./temp_dir/class_0/def.png', 0),
            #  ('./temp_dir/class_1/ghi.jpeg', 1), ('./temp_dir/class_1/jkl.png', 1),
            #  ('./temp_dir/class_1/mno/pqr.jpeg', 1), ('./temp_dir/class_1/mno/stu.jpg', 1)]
            print(data_folder_1.targets)
            # [0, 0, 1, 1, 1, 1]
            print(len(data_folder_1))
            # 6

            for i in range(len(data_folder_1)):
                img, label = data_folder_1[i]
                # do something with img and label
                print(type(img), img.size, label)
                # <class 'PIL.Image.Image'> (32, 32) 0


            transform = T.Compose(
                [
                    T.Resize(64),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        to_rgb=True,
                    ),
                ]
            )

            data_folder_2 = DatasetFolder(
                fake_data_dir,
                loader=lambda x: cv2.imread(x),  # load image with OpenCV
                extensions=(".jpg",),  # only load *.jpg files
                transform=transform,  # apply transform to every image
            )

            print([img_path for img_path, label in data_folder_2.samples])
            # ['./temp_dir/class_0/abc.jpg', './temp_dir/class_1/mno/stu.jpg']
            print(len(data_folder_2))
            # 2

            for img, label in iter(data_folder_2):
                # do something with img and label
                print(type(img), img.shape, label)
                # <class 'paddle.Tensor'> [3, 64, 64] 0

            shutil.rmtree(fake_data_dir)
    """

    def __init__(self,
                 root,
                 loader=None,
                 extensions=None,
                 transform=None,
                 is_valid_file=None):
        self.root = root
        self.transform = transform
        if extensions is None:
            extensions = IMG_EXTENSIONS
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions,
                               is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 directories in subfolders of: " +
                                self.root + "\n"
                                "Supported extensions are: " +
                                ",".join(extensions)))

        self.loader = default_loader if loader is None else loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.dtype = paddle.get_default_dtype()

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), 
                    and class_to_idx is a dictionary.

        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d for d in os.listdir(dir)
                if os.path.isdir(os.path.join(dir, d))
            ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def cv2_loader(path):
    cv2 = try_import('cv2')
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def default_loader(path):
    from paddle.vision import get_image_backend
    if get_image_backend() == 'cv2':
        return cv2_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(Dataset):
    """A generic data loader where the samples are arranged in this way:

    .. code-block:: text

        root/1.ext
        root/2.ext
        root/sub_dir/3.ext

    Args:
        root (str): Root directory path.
        loader (Callable, optional): A function to load a sample given its path. Default: None.
        extensions (list[str]|tuple[str], optional): A list of allowed extensions.
            Both :attr:`extensions` and :attr:`is_valid_file` should not be passed.
            If this value is not set, the default is to use ('.jpg', '.jpeg', '.png',
            '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'). Default: None.
        transform (Callable, optional): A function/transform that takes in
            a sample and returns a transformed version. Default: None.
        is_valid_file (Callable, optional): A function that takes path of a file
            and check if the file is a valid file. Both :attr:`extensions` and
            :attr:`is_valid_file` should not be passed. Default: None.

    Returns:
        :ref:`api_paddle_io_Dataset`. An instance of ImageFolder.

    Attributes:
        samples (list[str]): List of sample path.

    Example:

        .. code-block:: python

            import shutil
            import tempfile
            import cv2
            import numpy as np
            import paddle.vision.transforms as T
            from pathlib import Path
            from paddle.vision.datasets import ImageFolder


            def make_fake_file(img_path: str):
                if img_path.endswith((".jpg", ".png", ".jpeg")):
                    fake_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                    cv2.imwrite(img_path, fake_img)
                elif img_path.endswith(".txt"):
                    with open(img_path, "w") as f:
                        f.write("This is a fake file.")

            def make_directory(root, directory_hierarchy, file_maker=make_fake_file):
                root = Path(root)
                root.mkdir(parents=True, exist_ok=True)
                for subpath in directory_hierarchy:
                    if isinstance(subpath, str):
                        filepath = root / subpath
                        file_maker(str(filepath))
                    else:
                        dirname = list(subpath.keys())[0]
                        make_directory(root / dirname, subpath[dirname])

            directory_hirerarchy = [
                "abc.jpg",
                "def.png",
                {"ghi": [
                    "jkl.jpeg",
                    {"mno": [
                        "pqr.jpg"]}]},
                "this_will_be_ignored.txt",
            ]

            # You can replace this with any directory to explore the structure
            # of generated data. e.g. fake_data_dir = "./temp_dir"
            fake_data_dir = tempfile.mkdtemp()
            make_directory(fake_data_dir, directory_hirerarchy)
            image_folder_1 = ImageFolder(fake_data_dir)
            print(image_folder_1.samples)
            # ['./temp_dir/abc.jpg', './temp_dir/def.png',
            #  './temp_dir/ghi/jkl.jpeg', './temp_dir/ghi/mno/pqr.jpg']
            print(len(image_folder_1))
            # 4

            for i in range(len(image_folder_1)):
                (img,) = image_folder_1[i]
                # do something with img
                print(type(img), img.size)
                # <class 'PIL.Image.Image'> (32, 32)


            transform = T.Compose(
                [
                    T.Resize(64),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        to_rgb=True,
                    ),
                ]
            )

            image_folder_2 = ImageFolder(
                fake_data_dir,
                loader=lambda x: cv2.imread(x),  # load image with OpenCV
                extensions=(".jpg",),  # only load *.jpg files
                transform=transform,  # apply transform to every image
            )

            print(image_folder_2.samples)
            # ['./temp_dir/abc.jpg', './temp_dir/ghi/mno/pqr.jpg']
            print(len(image_folder_2))
            # 2

            for (img,) in iter(image_folder_2):
                # do something with img
                print(type(img), img.shape)
                # <class 'paddle.Tensor'> [3, 64, 64]

            shutil.rmtree(fake_data_dir)
     """

    def __init__(self,
                 root,
                 loader=None,
                 extensions=None,
                 transform=None,
                 is_valid_file=None):
        self.root = root
        if extensions is None:
            extensions = IMG_EXTENSIONS

        samples = []
        path = os.path.expanduser(root)

        if extensions is not None:

            def is_valid_file(x):
                return has_valid_extension(x, extensions)

        for root, _, fnames in sorted(os.walk(path, followlinks=True)):
            for fname in sorted(fnames):
                f = os.path.join(root, fname)
                if is_valid_file(f):
                    samples.append(f)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root +
                                "\n"
                                "Supported extensions are: " +
                                ",".join(extensions)))

        self.loader = default_loader if loader is None else loader
        self.extensions = extensions
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample of specific index.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return [sample]

    def __len__(self):
        return len(self.samples)

import os
import paddle

from paddle.vision import transforms

from util.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from util.data.transforms import RandomResizedCropAndInterpolationWithTwoPic
from util.data import create_transform

from dall_e.utils import map_pixels
from packages.masking_generator import MaskingGenerator, RandomMaskingGenerator
from packages.dataset_folder import DatasetFolder

ADE_DEFAULT_MEAN = (0.48897026, 0.46548377, 0.42939525)
ADE_DEFAULT_STD = (0.22846712, 0.22941928, 0.24038891)


class DataAugmentationForCAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = ADE_DEFAULT_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = ADE_DEFAULT_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if args.color_jitter > 0:
            self.common_transform = transforms.Compose([
                transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter),
                transforms.RandomHorizontalFlip(prob=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=args.second_input_size,
                    interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
                    scale=(args.crop_min_size, args.crop_max_size),
                ),
            ])
        else:
            self.common_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(prob=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=args.second_input_size,
                    interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
                    scale=(args.crop_min_size, args.crop_max_size),
                ),
            ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std)
            ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()

        if args.mask_generator == 'block':
            self.masked_position_generator = MaskingGenerator(
                args.window_size, num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block,
            )
        elif args.mask_generator == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, ratio_masking_patches=args.ratio_mask_patches
            )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)

        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForCAE,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class DataAugmentationForCAESingle(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = ADE_DEFAULT_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = ADE_DEFAULT_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if args.color_jitter > 0:
            self.common_transform = transforms.Compose([
                transforms.ColorJitter(args.color_jitter, args.color_jitter,
                                       args.color_jitter),
                transforms.RandomHorizontalFlip(prob=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size,
                    second_size=None,
                    interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
                    scale=(args.crop_min_size, args.crop_max_size),
                ),
            ])
        else:
            self.common_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(prob=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=None,
                    interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
                    scale=(args.crop_min_size, args.crop_max_size),
                ),
            ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std)
            ])

        if args.mask_generator == 'block':
            self.masked_position_generator = MaskingGenerator(
                args.window_size, num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block,
            )
        elif args.mask_generator == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, ratio_masking_patches=args.ratio_mask_patches
            )

        self.args = args
        if args.target_mode == 'random':
            self.vocab = paddle.load(args.target_path)['random_vector']
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __call__(self, image):
        for_patches = self.common_transform(image)
        transformed_patch = self.patch_transform(for_patches)

        if self.args.target_mode == 'random':
            normalized_img = self.visual_token_transform(for_patches)
            normalized_img = normalized_img.reshape([1, 3, self.args.window_size[0], self.args.patch_size[0], self.args.window_size[1], self.args.patch_size[1]])
            normalized_img = normalized_img.transpose([0, 2, 4, 1, 3, 5]).reshape([self.args.window_size[0] * self.args.window_size[1], -1])
            similarity = paddle.mm(normalized_img, self.vocab)
            # print(similarity.shape)     # (196, 8192)
            similarity = similarity.astype(paddle.float32)  # TODO fix amp bug
            target_id = similarity.argmax(axis=-1)
            return \
                transformed_patch, target_id, \
                self.masked_position_generator()
        else:
            return \
                transformed_patch, paddle.zeros_like(transformed_patch), \
                self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForCAE,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_cae_pretraining_dataset(args):
    low_case_path = args.data_path.lower()
    if 'imagenet' in low_case_path:
        if args.target_mode == 'clusterID':
            transform = DataAugmentationForCAE(args)
        elif args.target_mode == 'rgb' or args.target_mode == 'random':
            transform = DataAugmentationForCAESingle(args)
        print("Data Aug = %s" % str(transform))
        return DatasetFolder(args.data_path, transform=transform)
    elif 'ade' in low_case_path:
        raise NotImplementedError
        print('use ADE as the dataset.')
        from packages.ade20k_dataloader import ADE20K_Loader
        if args.target_mode == 'clusterID':
            transform = DataAugmentationForCAE(args)
        elif args.target_mode == 'rgb' or args.target_mode == 'random':
            transform = DataAugmentationForCAESingle(args)
        print("Data Aug = %s" % str(transform))
        return ADE20K_Loader(args.data_path, transform=transform)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        mode = 'train' if is_train else 'test'
        dataset = paddle.vision.datasets.Cifar100(args.data_path, mode=mode, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = DatasetFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = DatasetFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation='bicubic'),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
