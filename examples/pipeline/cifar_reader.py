# CIFAR10 reader

from __future__ import print_function
from __future__ import division

import itertools
import numpy
import tarfile
import six
from six.moves import cPickle as pickle

def reader_creator(filename,
                   sub_name,
                   cycle=False,
                   trainer_id=0,
                   num_trainers=1):
    def read_batch(batch):
        data = batch[six.b('data')]
        labels = batch.get(
            six.b('labels'), batch.get(six.b('fine_labels'), None))
        assert labels is not None
        for sample, label in six.moves.zip(data, labels):
            yield (sample / 255.0).astype(numpy.float32), int(label)

    def reader():
        cnt = 0
        while True:
            with tarfile.open(filename, mode='r') as f:
                names = (each_item.name for each_item in f
                         if sub_name in each_item.name)

                for name in names:
                    if six.PY2:
                        batch = pickle.load(f.extractfile(name))
                    else:
                        batch = pickle.load(
                            f.extractfile(name), encoding='bytes')
                    for item in read_batch(batch):
                        if cnt % num_trainers == trainer_id:
                            yield item
                        cnt += 1

            if not cycle:
                break
        print("cnt:", cnt)

    return reader

def train10(filepath, cycle=False, trainer_id=0, num_trainers=1):
    """
    CIFAR-10 training set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].

    :param cycle: whether to cycle through the dataset
    :type cycle: bool
    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(
        filepath,
        'data_batch',
        cycle=cycle,
        trainer_id=trainer_id,
        num_trainers=num_trainers)


def test10(filepath, cycle=False):
    """
    CIFAR-10 test set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].

    :param cycle: whether to cycle through the dataset
    :type cycle: bool
    :return: Test reader creator.
    :rtype: callable
    """
    return reader_creator(
        filepath,
        'test_batch',
        cycle=cycle)
