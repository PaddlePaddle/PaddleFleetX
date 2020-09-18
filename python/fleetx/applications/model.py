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
import sysconfig
import paddle.distributed.fleet as fleet
from fleetx.dataset.image_dataset import image_dataloader_from_filelist
from fleetx.dataset.bert_dataset import load_bert_dataset
from fleetx.dataset.transformer_dataset import transformer_data_generator
from fleetx.version import fleetx_version
from fleetx.dataset.ctr_data_generator import get_dataloader
from fleetx import utils


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

    def load_params(self, file_path):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        fluid.io.load_params(exe, file_path)

    def save_params(self, target_path):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        fluid.io.save_params(exe, target_path)


def download_model(fleet_path, model_name):
    version = fleetx_version.replace('-', '')
    if utils.is_first_worker():
        if not os.path.exists(fleet_path + model_name):
            if not os.path.exists(fleet_path + model_name + '.tar.gz'):
                os.system(
                    'wget -P {} --no-check-certificate https://fleet.bj.bcebos.com/models/{}/{}.tar.gz'.
                    format(fleet_path, version, model_name))
            os.system('tar -xf {}{}.tar.gz -C {}'.format(
                fleet_path, model_name, fleet_path))
    else:
        time.sleep(3)


class Resnet50(ModelBase):
    def __init__(self, data_layout='NCHW'):
        self.data_layout = data_layout
        super(Resnet50, self).__init__()
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        if data_layout == 'NCHW':
            model_name = 'resnet50_nchw'
        else:
            model_name = 'resnet50_nhwc'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

    def load_imagenet_from_file(self,
                                filelist,
                                batch_size=32,
                                phase='train',
                                shuffle=True,
                                use_dali=False):
        if phase != 'train':
            shuffle = False
        self.use_dali = use_dali
        data_layout = self.data_layout
        return image_dataloader_from_filelist(
            filelist,
            self.inputs,
            batch_size,
            phase,
            shuffle,
            use_dali,
            data_layout=data_layout)


class VGG16(ModelBase):
    def __init__(self, data_layout='NCHW'):
        super(VGG16, self).__init__()
        self.data_layout = data_layout
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        model_name = 'vgg16'
        if data_layout == 'NCHW':
            model_name = 'vgg16_nchw'
        else:
            model_name = 'vgg16_nhwc'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target
        self.use_dali = False

    def load_imagenet_from_file(self,
                                filelist,
                                batch_size=32,
                                phase='train',
                                shuffle=True,
                                use_dali=False):
        if phase != 'train':
            shuffle = False
        self.use_dali = use_dali
        data_layout = self.data_layout
        return image_dataloader_from_filelist(
            filelist,
            self.inputs,
            batch_size,
            phase,
            shuffle,
            use_dali,
            data_layout=data_layout)


class Transformer(ModelBase):
    def __init__(self):
        super(Transformer, self).__init__()
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        model_name = 'transformer'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

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


class BertLarge(ModelBase):
    def __init__(self, lang='ch'):
        super(BertLarge, self).__init__()
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        if lang == 'ch':
            model_name = 'bert_large'
        else:
            model_name = 'bert_large_en'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

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


class BertHuge(ModelBase):
    def __init__(self):
        super(BertHuge, self).__init__()
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        model_name = 'bert_huge'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

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


class BertGiant(ModelBase):
    def __init__(self):
        super(BertGiant, self).__init__()
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        model_name = 'bert_giant'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

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


class BertBase(ModelBase):
    def __init__(self, lang='ch'):
        super(BertBase, self).__init__()
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        if lang == 'ch':
            model_name = 'bert_base'
        else:
            model_name = 'bert_base_en'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

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


class MultiSlotCTR(ModelBase):
    def __init__(self):
        super(MultiSlotCTR, self).__init__()
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        model_name = 'ctr'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

    def load_criteo_from_file(self,
                              train_files_path,
                              sparse_feature_dim=1000001,
                              batch_size=1000,
                              shuffle=True):
        return get_dataloader(
            self.inputs,
            train_files_path,
            sparse_feature_dim=sparse_feature_dim,
            batch_size=batch_size,
            shuffle=shuffle)


class Resnet50Mlperf(ModelBase):
    def __init__(self):
        self.data_layout = "NHWC"
        model_name = 'resnet50_mlperf'
        super(Resnet50Mlperf, self).__init__()
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

    def load_imagenet_from_file(self,
                                filelist,
                                batch_size=32,
                                phase='train',
                                shuffle=True,
                                use_dali=False):
        if phase != 'train':
            shuffle = False
        self.use_dali = use_dali
        data_layout = self.data_layout
        return image_dataloader_from_filelist(
            filelist,
            self.inputs,
            batch_size,
            phase,
            shuffle,
            use_dali,
            data_layout=data_layout)                 
