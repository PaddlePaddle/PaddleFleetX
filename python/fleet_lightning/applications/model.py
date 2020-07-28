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
import time
from .util import *
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from fleet_lightning.dataset.image_dataset import image_dataloader_from_filelist
from fleet_lightning.dataset.bert_dataset import load_bert_dataset
from fleet_lightning.dataset.transformer_dataset import transformer_data_generator


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
        gpu_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
        if gpu_id == 0:
            if not os.path.exists('resnet50'):
                if not os.path.exists('resnet50.tar.gz'):
                    os.system(
                        'wget --no-check-certificate https://fleet.bj.bcebos.com/models/{}.tar.gz'.
                        format('resnet50'))
                os.system('tar -xf {}.tar.gz'.format('resnet50'))
        else:
            time.sleep(3)
        inputs, loss, startup, main, unique_generator, checkponits = load_program(
            "resnet50")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkponits

    def load_imagenet_from_file(self,
                                filelist,
                                batch_size=32,
                                phase='train',
                                shuffle=True,
                                use_dali=False):
        return image_dataloader_from_filelist(
            filelist, self.inputs, batch_size, phase, shuffle, use_dali)


class VGG16(ModelBase):
    def __init__(self):
        super(VGG16, self).__init__()
        gpu_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
        if gpu_id == 0:
            if not os.path.exists('vgg16'):
                os.system(
                    'wget --no-check-certificate https://fleet.bj.bcebos.com/models/{}.tar.gz'.
                    format('vgg16'))
                os.system('tar -xf {}.tar.gz'.format('vgg16'))
        else:
            time.sleep(3)
        inputs, loss, startup, main, unique_generator, checkponits = load_program(
            "vgg16")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkponits

    def load_imagenet_from_file(self,
                                filelist,
                                batch_size=32,
                                phase='train',
                                shuffle=True,
                                use_dali=False):
        return image_dataloader_from_filelist(
            filelist, self.inputs, batch_size, phase, shuffle, use_dali)


class Transformer(ModelBase):
    def __init__(self):
        super(Transformer, self).__init__()
        gpu_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
        if gpu_id == 0:
            if not os.path.exists('transformer'):
                os.system(
                    'wget --no-check-certificate https://fleet.bj.bcebos.com/models/{}.tar.gz'.
                    format('transformer'))
                os.system('tar -xf {}.tar.gz'.format('transformer'))
        else:
            time.sleep(3)
        inputs, loss, startup, main, unique_generator, checkponits = load_program(
            "transformer")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkponits

    def load_wmt16_dataset_from_file(self,
                                     src_vocab_fpath,
                                     trg_vocab_fpath,
                                     train_file_pattern,
                                     batch_size=2048,
                                     shuffle=True):
        return transformer_data_generator(
            src_vocab_fpath,
            trg_vocab_fpath,
            train_file_pattern,
            inputs=self.inputs,
            batch_size=batch_size,
            shuffle=shuffle)


class Bert_large(ModelBase):
    def __init__(self):
        super(Bert_large, self).__init__()
        gpu_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
        if gpu_id == 0:
            if not os.path.exists('bert_large'):
                if not os.path.exists('bert_large.tar.gz'):
                    os.system(
                        'wget --no-check-certificate https://fleet.bj.bcebos.com/models/{}.tar.gz'.
                        format('bert_large'))
                os.system('tar -xf {}.tar.gz'.format('bert_large'))
        else:
            time.sleep(3)
        inputs, loss, startup, main, unique_generator, checkponits = load_program(
            "bert_large")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkponits

    def load_digital_dataset_from_file(self,
                                       data_dir,
                                       vocab_path,
                                       batch_size=16,
                                       max_seq_len=128,
                                       in_tokens=False):
        return load_bert_dataset(
            data_dir,
            vocab_path,
            inputs=self.inputs,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            in_tokens=in_tokens)


class Bert_base(ModelBase):
    def __init__(self):
        super(Bert_base, self).__init__()
        gpu_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
        if gpu_id == 0:
            if not os.path.exists('bert_base'):
                if not os.path.exists('bert_base.tar.gz'):
                    os.system(
                        'wget --no-check-certificate https://fleet.bj.bcebos.com/models/{}.tar.gz'.
                        format('bert_base'))
                os.system('tar -xf {}.tar.gz'.format('bert_base'))
        else:
            time.sleep(3)
        inputs, loss, startup, main, unique_generator, checkponits = load_program(
            "bert_base")
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkponits

    def load_digital_dataset_from_file(self,
                                       data_dir,
                                       vocab_path,
                                       batch_size=4096,
                                       max_seq_len=512,
                                       in_tokens=True):
        return load_bert_dataset(
            data_dir,
            vocab_path,
            inputs=self.inputs,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            in_tokens=in_tokens)
