# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import copy
import logging

import paddle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISTILLATION_LOSS_TYPES = [
    'mse', 'kl_div', 'token_wise_contrastive'
]

_distillation_config_default = {

    # distillation loss type, default is mse and mse loss function is used
    'distillation_loss_type': 'mse',
    # positive sample knowledge distillationn, default is False
    # 'positive_sample_kd': False,
    # distillation loss value, default is 1.0
    'distill_loss_ratio': 1.0,
    # temperature T, is used to reduce magnitude difference among the class likelihood values, default is 1.0
    'T': 1.0, 
}


def _parse_configs(user_config):
    """
    check if user's configs are valid.
    Args:
        user_config(dict): user's config.
    Return:
        configs(dict): final configs will be used.
    """

    configs = copy.deepcopy(_distillation_config_default)
    configs.update(user_config)

    # check if configs is valid

    assert configs['distillation_loss_type'] in DISTILLATION_LOSS_TYPES or configs['distillation_loss_type'] is None, \
        "Unknown distillation_loss_type: {}. only supports {} ".format(configs['distillation_loss_type'],
                DISTILLATION_LOSS_TYPES)

    assert isinstance(configs['T'], float), \
        "temperature must be float value."
    
    assert isinstance(configs['distill_loss_ratio'], float), \
        "ratio of distillation loss must be float value."

    return configs

def parse_teacher_cfg(configs):
    if 'Teacher' not in configs:
        configs['Teacher'] = dict()
        configs['Teacher']['Model'] = configs['StudentModel']
        logger.warning("No teacher model, use student model as teacher model.")

    cfg_teacher = configs['Teacher']
    if 'save_load' not in cfg_teacher:
        cfg_teacher['save_load'] = dict()
        cfg_teacher['save_load']['ckpt_dir'] = configs['Student_ckpt']
        logger.warning("No teacher ckpt, use student model ckpt instead.")

    assert isinstance(cfg_teacher['save_load']['ckpt_dir'] , str), \
        "teacher model path must be string value."
    
    return configs


class Distillation(object):
    """
    Knowledge distillation(kd): Distill the knowledge from teacher model
    into student model
    """
    def __init__(self,configs=None):
        if configs is None:
            configs = _distillation_config_default
        else:
    	    assert isinstance(configs, dict), "configs must be dict"

        configs = _parse_configs(configs)

        self.configs = configs

        self._get_loss_fn()

    def _get_loss_fn(self):
        """
        Get the knowledge distillation loss function type
        """

        if self.configs['distillation_loss_type'] == 'mse':
        	self.loss_fn = paddle.nn.functional.mse_loss

        if self.configs['distillation_loss_type'] == 'kl_div':
        	self.loss_fn = paddle.nn.functional.kl_div

        if self.configs['distillation_loss_type'] == 'token_wise_contrastive':
        	self.loss_fn = contrastive_loss

    def distill(self, student_logit, teacher_logit):
        print(self.configs['distillation_loss_type'])
        return self.configs['distill_loss_ratio'] * self.loss_fn(student_logit, teacher_logit)


def contrastive_loss(a, b):
    const_loss = 0.0

    for i in range(a.shape[0]):
        w12 = paddle.matmul(a[i], b[i].T)
        eps=1e-8

        w1 = paddle.sum(paddle.multiply(a[i], a[i]), axis=1, keepdim=True)
        w2 = paddle.sum(paddle.multiply(b[i].T, b[i].T), axis=0, keepdim=True)
        n12 = paddle.sqrt(paddle.clip(paddle.matmul(w1, w2), min=eps * eps))
        cos_sim = w12 / n12

        pos = paddle.diag(cos_sim)
        neg = paddle.sum(cos_sim - paddle.nn.functional.diag_embed(pos), axis=1) # [1, 1]
        s = paddle.exp(pos / neg)
        loss = - paddle.sum(paddle.log(s)) 

        const_loss = const_loss + loss

    return const_loss / a.shape[0]

