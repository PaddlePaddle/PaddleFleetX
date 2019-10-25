import paddle
import paddle.fluid as fluid
import math
import os
import numpy as np
from paddle.fluid import unique_name
from paddle.fluid.layers import dist_algo


__all__ = ["ResNet_ARCFACE", "ResNet_ARCFACE50", "ResNet_ARCFACE101",
           "ResNet_ARCFACE152"]


train_parameters = {
    "input_size": [3, 112, 112],
    "input_mean": [127.5, 127.5, 127.5],
    "input_std": [128.0, 128.0, 128.0],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 128,
        "epochs": [100000, 160000, 220000],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class ResNet_ARCFACE():
    def __init__(self, layers=50):
        self.params = train_parameters
        self.layers = layers

    def net(self,
            input,
            label,
            emb_dim=512,
            class_dim=85742,
            loss_type='arcface',
            margin=0.5,
            scale=64.0):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {}, but input layer is {}".format(
            supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 14, 3]
            num_filters = [64, 128, 256, 512]
        elif layers == 101:
            depth = [3, 4, 23, 3]
            num_filters = [256, 512, 1024, 2048]
        elif layers == 152:
            depth = [3, 8, 36, 3]
            num_filters = [256, 512, 1024, 2048]

        conv = self.conv_bn_layer(
            input=input, num_filters=64, filter_size=3, stride=1,
            pad=1, act='prelu')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 else 1)

        bn = fluid.layers.batch_norm(input=conv, act=None, epsilon=2e-05)
        drop = fluid.layers.dropout(x=bn, dropout_prob=0.4,
            dropout_implementation='upscale_in_train')
        fc = fluid.layers.fc(
            input=drop,
            size=emb_dim,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False, fan_in=0.0)),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer()))
        emb = fluid.layers.batch_norm(input=fc, act=None, epsilon=2e-05)

        nranks = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
        rank_id = int(os.getenv("PADDLE_TRAINER_ID", 0))

        if loss_type == 'softmax':
            loss = self.fc_classify(emb, label, class_dim)
        elif loss_type == 'arcface':
            loss = self.arcface(emb, label, class_dim)
        elif loss_type == 'dist_softmax':
            loss = dist_algo._distributed_softmax_classify(
                x=emb, label=label, class_num=class_dim, nranks=nranks, rank_id=rank_id)
        elif loss_type == 'dist_arcface':
            loss = dist_algo._distributed_arcface_classify(
                x=emb, label=label, class_num=class_dim, nranks=nranks, rank_id=rank_id,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(scale=0.01)))
        else:
            raise ValueError('Invalid loss type:', loss_type)

        return emb, loss

    def fc_classify(self, input, label, out_dim):
        stdv = 1.0 / math.sqrt(input.shape[1] * 1.0)
        out = fluid.layers.fc(input=input,
                              size=out_dim,
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv,
                                                                        stdv)))
        loss, prob = fluid.layers.softmax_with_cross_entropy(logits=out, 
            label=label, return_softmax=True)
        avg_loss = fluid.layers.mean(x=loss)
        return avg_loss, prob

    def arcface(self, input, label, out_dim):
        input_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(input), dim=1))
        input = fluid.layers.elementwise_div(input, input_norm, axis=0)

        weight = fluid.layers.create_parameter(
                    shape=[input.shape[1], out_dim], 
                    dtype='float32',
                    name=unique_name.generate('final_fc_w'),
                    attr=fluid.param_attr.ParamAttr(
                        initializer=fluid.initializer.Xavier(uniform=False, fan_in=0.0)))

        weight_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(weight), dim=0))
        weight = fluid.layers.elementwise_div(weight, weight_norm, axis=1)
        cos = fluid.layers.mul(input, weight)

        theta = fluid.layers.acos(cos)
        margin_cos = fluid.layers.cos(theta + 0.5)
        one_hot = fluid.layers.one_hot(label, out_dim)
        diff = (margin_cos - cos) * one_hot
        target_cos = cos + diff
        logit = fluid.layers.scale(target_cos, scale=64.)

        loss, prob = fluid.layers.softmax_with_cross_entropy(logits=logit, 
            label=label, return_softmax=True)
        avg_loss = fluid.layers.mean(x=loss)

        one_hot.stop_gradient = True

        return avg_loss, prob

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      pad=0,
                      groups=1,
                      act=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=pad,
            groups=groups,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False, fan_in=0.0)),
            bias_attr=False)
        if act == 'prelu':
            bn = fluid.layers.batch_norm(input=conv, act=None, epsilon=2e-05,
                momentum=0.9)
            return fluid.layers.leaky_relu(x=bn, alpha=0.25)
        else:
            return fluid.layers.batch_norm(input=conv, act=act, epsilon=2e-05)

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride):
        if self.layers < 101:
            bn1 = fluid.layers.batch_norm(input=input, act=None, epsilon=2e-05)
            conv1 = self.conv_bn_layer(
                input=bn1, num_filters=num_filters, filter_size=3, pad=1, act='prelu')
            conv2 = self.conv_bn_layer(
                input=conv1, num_filters=num_filters, filter_size=3, stride=stride, pad=1, act=None)
        else:
            bn0 = fluid.layers.batch_norm(input=input, act=None, epsilon=2e-05)
            conv0 = self.conv_bn_layer(
                input=bn0, num_filters=num_filters/4, filter_size=1, pad=0, act='prelu')
            conv1 = self.conv_bn_layer(
                input=conv0, num_filters=num_filters/4, filter_size=3, pad=1, act='prelu')
            conv2 = self.conv_bn_layer(
                input=conv1, num_filters=num_filters, filter_size=1, stride=stride, pad=0, act=None)

        short = self.shortcut(input, num_filters, stride)
        return fluid.layers.elementwise_add(x=short, y=conv2, act=None)


def ResNet_ARCFACE50():
    model = ResNet_ARCFACE(layers=50)
    return model


def ResNet_ARCFACE101():
    model = ResNet_ARCFACE(layers=101)
    return model


def ResNet_ARCFACE152():
    model = ResNet_ARCFACE(layers=152)
    return model
