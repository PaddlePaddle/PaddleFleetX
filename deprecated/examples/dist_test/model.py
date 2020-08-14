# -*- coding: utf-8 -*-
import paddle.fluid as fluid


def build_train_net():
    '''
    Create training network
    '''
    return _build_net(False)


def build_test_net():
    '''
    Create testing network
    '''
    return _build_net(True)


def _build_net(is_test):
    '''
    Build MLP network based on MNIST dataset
    '''
    with fluid.unique_name.guard():
        image = fluid.layers.data(name='image', shape=[784], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        fc0 = fluid.layers.fc(image, size=128, act='relu')
        CLASS_NUM = 10
        fc1 = fluid.layers.fc(fc0, size=CLASS_NUM)

        if is_test:
            softmax = fluid.layers.softmax(fc1)
            acc = fluid.layers.accuracy(softmax, label=label, k=1)
            return [image, label], [acc]
        else:
            cross_entropy = fluid.layers.softmax_with_cross_entropy(fc1, label)
            loss = fluid.layers.reduce_mean(cross_entropy)
            return [image, label], [loss]
