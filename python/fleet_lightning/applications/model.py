#!/usr/bin/python
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from .util import *
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from fleet_lightning.dataset.image_dataset import image_dataloader_from_filelist
from fleet_lightning.dataset.bert_dataset import load_bert_dataset
from fleet_lightning.dataset.translation_dataset import prepare_data_generator, prepare_feed_dict_list


class ModelBase(object):
    def __init__(self):
        self.inputs = []
        self.startup_prog = None
        self.main_prog = None

    def inputs(self):
        return self.inputs

    def get_loss(self):
        return self.loss

    def hidden(self):
        return []

    def parameter_list(self):
        return self.main_prog.all_parameters()

    def startup_program(self):
        return self.startup_prog

    def main_program(self):
        return self.main_prog


class Resnet50(ModelBase):
    def __init__(self):
        super(Resnet50, self).__init__()
        if not os.path.exists('resnet50'):
            os.system(
                'wget --no-check-certificate https://fleet.bj.bcebos.com/models/{}.tar.gz'.
                format('resnet50'))
            os.system('tar -xf {}.tar.gz'.format('resnet50'))
        inputs, loss, startup, main, unique_generator = load_program(
            "resnet50")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss

    def load_imagenet_from_file(self,
                                filelist,
                                phase='train',
                                shuffle=True,
                                use_dali=False):
        return image_dataloader_from_filelist(filelist, self.inputs, phase,
                                              shuffle, use_dali)


class VGG16(ModelBase):
    def __init__(self):
        super(VGG16, self).__init__()
        if not os.path.exists('vgg16'):
            os.system(
                'wget --no-check-certificate https://fleet.bj.bcebos.com/models/{}.tar.gz'.
                format('vgg16'))
            os.system('tar -xf {}.tar.gz'.format('vgg16'))
        inputs, loss, startup, main, unique_generator = load_program("vgg16")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss

    def load_imagenet_from_file(self,
                                filelist,
                                phase='train',
                                shuffle=True,
                                use_dali=False):
        return image_dataloader_from_filelist(filelist, self.inputs, phase,
                                              shuffle, use_dali)


class Transformer(ModelBase):
    def __init__(self):
        super(Transformer, self).__init__()
        if not os.path.exists('transformer'):
            os.system(
                'wget --no-check-certificate https://fleet.bj.bcebos.com/models/{}.tar.gz'.
                format('transformer'))
            os.system('tar -xf {}.tar.gz'.format('transformer'))
        inputs, loss, startup, main, unique_generator = load_program(
            "transformer")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss

    def load_wmt16_dataset_from_file(self,
                                     src_vocab_fpath,
                                     trg_vocab_fpath,
                                     train_file_pattern,
                                     batch_size=4096,
                                     shuffle=True):
        return prepare_data_generator(src_vocab_fpath, trg_vocab_fpath,
                                      train_file_pattern, batch_size, shuffle)

    def generate_feed_dict_list(self, data_reader):
        input_name = []
        for item in self.inputs:
            input_name.append(item.name)
        return prepare_feed_dict_list(data_reader, input_name)


class Bert(ModelBase):
    def __init__(self):
        super(Bert, self).__init__()
        if not os.path.exists('bert'):
            os.system(
                'wget --no-check-certificate https://fleet.bj.bcebos.com/models/{}.tar.gz'.
                format('bert'))
            os.system('tar -xf {}.tar.gz'.format('bert'))
        inputs, loss, startup, main, unique_generator = load_program("bert")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss

    def load_digital_dataset_from_file(self,
                                       data_dir,
                                       vocab_path,
                                       batch_size=8196,
                                       max_seq_len=512):
        return load_bert_dataset(
            data_dir,
            vocab_path,
            inputs=self.inputs,
            batch_size=batch_size,
            max_seq_len=max_seq_len)


class Faster_rcnn(ModelBase):
    def __init__(self):
        super(Faster_rcnn, self).__init__()
        if not os.path.exists('faster_rcnn'):
            os.system(
                'wget --no-check-certificate https://fleet.bj.bcebos.com/models/faster_rcnn.tar.gz'
            )
            os.system('tar -xf faster_rcnn.tar.gz')
        inputs, loss, startup, main, unique_generator = load_program(
            "faster_rcnn")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
