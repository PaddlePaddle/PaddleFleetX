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
from fleetx.dataset.word2vec_dataset import load_w2v_dataset
from fleetx.version import fleetx_version
from fleetx.dataset.ctr_data_generator import get_dataloader
from fleetx import utils


class ModelBase(object):
    """
    Base class for loading models.

    After loading the model we saved, you can get the following info of a model: 
    
    main_program, startup_program, loss, inputs, checkpoints for recompute, etc. 
    """

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
    """
    Download pre-saved model if it does not exist in your local path. 
    """
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
        while not os.path.exists(fleet_path + model_name):
            time.sleep(3)


class Resnet50(ModelBase):
    def __init__(self, data_layout='NCHW'):
        """
        Load pre-saved Resnet50 model. 

        Args:
            data_layout: data layout of image inputs. 
                         NCHW: [3, 224, 224]
                         NHWC: [224, 224, 3]

        Example:

        ..code:: python

            import paddle
            import fleetx as X
            import paddle.distributed.fleet as fleet
            paddle.enable_static()
            
            configs = X.parse_train_configs()
            fleet.init(is_collective=True)
            model = X.applications.Resnet50()
            optimizer = paddle.fluid.optimizer.Momentum(learning_rate=configs.lr)
            optimizer = fleet.distributed_optimizer(optimizer)
            optimizer.minimize(model.loss)
        """
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

    def get_train_dataloader(self,
                             local_path,
                             batch_size=32,
                             shuffle=True,
                             use_dali=False):
        """
        Load train imagenet data from local_path. 
        """
        filelist = local_path + '/train.txt'
        data_layout = self.data_layout
        return image_dataloader_from_filelist(
            filelist,
            self.inputs,
            batch_size,
            phase='train',
            shuffle=shuffle,
            use_dali=use_dali,
            data_layout=data_layout)

    def get_val_dataloader(self,
                           local_path,
                           batch_size=32,
                           shuffle=False,
                           use_dali=False):
        """
        Load val imagenet data from local_path. 
        """
        filelist = local_path + '/val.txt'
        data_layout = self.data_layout
        return image_dataloader_from_filelist(
            filelist,
            self.inputs,
            batch_size,
            phase='val',
            shuffle=shuffle,
            use_dali=use_dali,
            data_layout=data_layout)


class VGG16(ModelBase):
    def __init__(self, data_layout='NCHW'):
        """
        Load pre-saved VGG16 model.

        Args:
            data_layout: data layout of image inputs.
                         NCHW: [3, 224, 224]
                         NHWC: [224, 224, 3]

        Example:

        ..code:: python

            import paddle
            import fleetx as X
            import paddle.distributed.fleet as fleet
            paddle.enable_static()

            configs = X.parse_train_configs()
            fleet.init(is_collective=True)
            model = X.applications.VGG16()
            optimizer = paddle.fluid.optimizer.Momentum(learning_rate=configs.lr)
            optimizer = fleet.distributed_optimizer(optimizer)
            optimizer.minimize(model.loss)
        """
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

    def get_train_dataloader(self,
                             local_path,
                             batch_size=32,
                             shuffle=True,
                             use_dali=False):
        """
        Load train imagenet data from local_path.
        """
        filelist = local_path + '/train.txt'
        data_layout = self.data_layout
        return image_dataloader_from_filelist(
            filelist,
            self.inputs,
            batch_size,
            phase='train',
            shuffle=shuffle,
            use_dali=use_dali,
            data_layout=data_layout)

    def get_val_dataloader(self,
                           local_path,
                           batch_size=32,
                           shuffle=False,
                           use_dali=False):
        """
        Load val imagenet data from local_path.
        """
        filelist = local_path + '/val.txt'
        data_layout = self.data_layout
        return image_dataloader_from_filelist(
            filelist,
            self.inputs,
            batch_size,
            phase='val',
            shuffle=shuffle,
            use_dali=use_dali,
            data_layout=data_layout)


class Transformer(ModelBase):
    def __init__(self):
        """
        Load pre-saved Transformer model.

        Example:

        ..code:: python

            import paddle
            import fleetx as X
            import paddle.distributed.fleet as fleet
            paddle.enable_static()

            configs = X.parse_train_configs()
            fleet.init(is_collective=True)
            model = X.applications.Transformer()
            optimizer = paddle.fluid.optimizer.Adam(learning_rate=configs.lr)
            optimizer = fleet.distributed_optimizer(optimizer)
            optimizer.minimize(model.loss)
        """
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

    def get_train_dataloader(self,
                             src_vocab_fpath,
                             trg_vocab_fpath,
                             train_file_pattern,
                             batch_size=2048,
                             shuffle=True):
        """
        Load WMT data from local path. 
        """
        return transformer_data_generator(
            src_vocab_fpath,
            trg_vocab_fpath,
            train_file_pattern,
            inputs=self.inputs,
            batch_size=batch_size,
            shuffle=shuffle)


class BertLarge(ModelBase):
    def __init__(self, lang='ch'):
        """
        Load pre-saved BertLarge model.

        Args:
            lang: language of your training data, currently wo support chinese and english. 

        Example:

        ..code:: python

            import paddle
            import fleetx as X
            import paddle.distributed.fleet as fleet
            paddle.enable_static()

            configs = X.parse_train_configs()
            fleet.init(is_collective=True)
            model = X.applications.BertLarge()
            optimizer = paddle.fluid.optimizer.Adam(learning_rate=configs.lr)
            optimizer = fleet.distributed_optimizer(optimizer)
            optimizer.minimize(model.loss)
        """
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
        self.lang = lang
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

    def get_train_dataloader(self,
                             data_dir,
                             batch_size=4096,
                             max_seq_len=512,
                             in_tokens=True,
                             shuffle=True):
        """
        Load train Wiki data of language defined in model from local path.
        """
        if not in_tokens:
            batch_size = batch_size / max_seq_len
        return load_bert_dataset(
            data_dir,
            inputs=self.inputs,
            batch_size=batch_size,
            lang=self.lang,
            phase='train',
            max_seq_len=max_seq_len,
            in_tokens=in_tokens,
            shuffle=shuffle)

    def get_val_dataloader(self,
                           data_dir,
                           batch_size=4096,
                           max_seq_len=512,
                           in_tokens=True,
                           shuffle=False):
        """
        Load val Wiki data of language defined in model from local path.
        """
        if not in_tokens:
            batch_size = batch_size / max_seq_len
        return load_bert_dataset(
            data_dir,
            inputs=self.inputs,
            batch_size=batch_size,
            lang=self.lang,
            phase='val',
            max_seq_len=max_seq_len,
            in_tokens=in_tokens,
            shuffle=shuffle)


class BertHuge(ModelBase):
    def __init__(self, lang='en'):
        """
        Load pre-saved BertHuge model.

        Example:

        ..code:: python

            import paddle
            import fleetx as X
            import paddle.distributed.fleet as fleet
            paddle.enable_static()

            configs = X.parse_train_configs()
            fleet.init(is_collective=True)
            model = X.applications.BertHuge()
            optimizer = paddle.fluid.optimizer.Adam(learning_rate=configs.lr)
            optimizer = fleet.distributed_optimizer(optimizer)
            optimizer.minimize(model.loss)
        """
        super(BertHuge, self).__init__()
        if lang == 'ch':
            raise Exception(
                "English model is not supported currently in BertHuge")
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        model_name = 'bert_huge'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.lang = lang
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

    def get_train_dataloader(self,
                             data_dir,
                             batch_size=4096,
                             max_seq_len=512,
                             in_tokens=True,
                             shuffle=True):
        """
        Load train Wiki data of language defined in model from local path.
        """
        if not in_tokens:
            batch_size = batch_size / max_seq_len
        return load_bert_dataset(
            data_dir,
            inputs=self.inputs,
            batch_size=batch_size,
            lang=self.lang,
            phase='train',
            max_seq_len=max_seq_len,
            in_tokens=in_tokens,
            shuffle=shuffle)

    def get_val_dataloader(self,
                           data_dir,
                           batch_size=4096,
                           max_seq_len=512,
                           in_tokens=True,
                           shuffle=False):
        """
        Load val Wiki data of language defined in model from local path.
        """
        if not in_tokens:
            batch_size = batch_size / max_seq_len
        return load_bert_dataset(
            data_dir,
            inputs=self.inputs,
            batch_size=batch_size,
            lang=self.lang,
            phase='val',
            max_seq_len=max_seq_len,
            in_tokens=in_tokens,
            shuffle=shuffle)


class BertGiant(ModelBase):
    def __init__(self, lang='en'):
        """
        Load pre-saved BertGiant model.

        Example:

        ..code:: python

            import paddle
            import fleetx as X
            import paddle.distributed.fleet as fleet
            paddle.enable_static()

            configs = X.parse_train_configs()
            fleet.init(is_collective=True)
            model = X.applications.BertGiant()
            optimizer = paddle.fluid.optimizer.Adam(learning_rate=configs.lr)
            optimizer = fleet.distributed_optimizer(optimizer)
            optimizer.minimize(model.loss)
        """
        super(BertGiant, self).__init__()
        if lang == 'ch':
            raise Exception(
                "Chinese model is not supported currently in BertGiant")
        fleet_path = sysconfig.get_paths()["purelib"] + '/fleetx/applications/'
        model_name = 'bert_giant'
        download_model(fleet_path, model_name)
        inputs, loss, startup, main, unique_generator, checkpoints, target = load_program(
            fleet_path + model_name)
        self.startup_prog = startup
        self.main_prog = main
        self.lang = lang
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

    def get_train_dataloader(self,
                             data_dir,
                             batch_size=4096,
                             max_seq_len=512,
                             in_tokens=True,
                             shuffle=True):
        """
        Load train Wiki data of language defined in model from local path.
        """
        if not in_tokens:
            batch_size = batch_size / max_seq_len
        return load_bert_dataset(
            data_dir,
            inputs=self.inputs,
            batch_size=batch_size,
            lang=self.lang,
            phase='train',
            max_seq_len=max_seq_len,
            in_tokens=in_tokens,
            shuffle=shuffle)

    def get_val_dataloader(self,
                           data_dir,
                           batch_size=4096,
                           max_seq_len=512,
                           in_tokens=True,
                           shuffle=False):
        """
        Load val Wiki data of language defined in model from local path.
        """
        if not in_tokens:
            batch_size = batch_size / max_seq_len
        return load_bert_dataset(
            data_dir,
            inputs=self.inputs,
            batch_size=batch_size,
            lang=self.lang,
            phase='val',
            max_seq_len=max_seq_len,
            in_tokens=in_tokens,
            shuffle=shuffle)


class BertBase(ModelBase):
    def __init__(self, lang='ch'):
        """
        Load pre-saved BertBase model.

        Args:
            lang: language of your training data, currently wo support chinese and english.

        Example:

        ..code:: python

            import paddle
            import fleetx as X
            import paddle.distributed.fleet as fleet
            paddle.enable_static()

            configs = X.parse_train_configs()
            fleet.init(is_collective=True)
            model = X.applications.BertBase()
            optimizer = paddle.fluid.optimizer.Adam(learning_rate=configs.lr)
            optimizer = fleet.distributed_optimizer(optimizer)
            optimizer.minimize(model.loss)
        """
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
        self.lang = lang
        self.inputs = inputs
        self.loss = loss
        self.checkpoints = checkpoints
        self.target = target

    def get_train_dataloader(self,
                             data_dir,
                             batch_size=4096,
                             max_seq_len=512,
                             in_tokens=True,
                             shuffle=True):
        """
        Load train Wiki data of language defined in model from local path.
        """
        if not in_tokens:
            batch_size = int(batch_size / max_seq_len)

        return load_bert_dataset(
            data_dir,
            inputs=self.inputs,
            batch_size=batch_size,
            lang=self.lang,
            phase='train',
            max_seq_len=max_seq_len,
            in_tokens=in_tokens,
            shuffle=shuffle)

    def get_val_dataloader(self,
                           data_dir,
                           batch_size=4096,
                           max_seq_len=512,
                           in_tokens=True,
                           shuffle=False):
        """
        Load train Wiki data of language defined in model from local path.
        """
        if not in_tokens:
            batch_size = batch_size / max_seq_len
        return load_bert_dataset(
            data_dir,
            inputs=self.inputs,
            batch_size=batch_size,
            lang=self.lang,
            phase='val',
            max_seq_len=max_seq_len,
            in_tokens=in_tokens,
            shuffle=shuffle)


class MultiSlotCTR(ModelBase):
    def __init__(self):
        """
        Load pre-saved MultiSlotCTR model.

        Example:

        ..code:: python

            import paddle
            import fleetx as X
            import paddle.distributed.fleet as fleet
            paddle.enable_static()

            configs = X.parse_train_configs()
            fleet.init()
            model = X.applications.MultiSlotCTR()
            optimizer = paddle.fluid.optimizer.Adam(learning_rate=configs.lr)
            optimizer = fleet.distributed_optimizer(optimizer)
            optimizer.minimize(model.loss)
        """
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
        """
        Special resnet50 model prepared for MLPerf.
        """
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
