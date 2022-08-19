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

import copy


class GPTConfig(dict):
    def __init__(self, yaml_dict, **kw):
        super(GPTConfig, self).__init__(**kw)

        self.Global = {
            'device': 'gpu',
            'seed': 1024,
        }

        self.Engine = {
            'max_steps': 500000,
            'num_train_epochs': 1,
            'accumulate_steps': 1,
            'logging_freq': 1,
            'eval_freq': 500,
            'eval_iters': 10,
            'test_iters': None,
            'mix_precision': {
                'use_pure_fp16': True,
                'scale_loss': 32768.0,
                'custom_black_list': [
                    "reduce_sum", "c_softmax_with_cross_entropy",
                    "elementwise_div"
                ],
                'custom_white_list': ["lookup_table", "lookup_table_v2"],
            },
            'save_load': {
                'save_steps': 1000,
                'output_dir': None,
                'ckpt_dir': None,
            }
        }

        self.Data = {
            'batch_size': {
                'global_batch_size': None,
                'local_batch_size': None,
                'micro_batch_size': 8,
            },
            'dataset': {
                'input_dir': None,
                'split': '949,50,1',
                'max_seq_len': 1024,
            }
        }

        self.Model = {
            'vocab_size': 50304,
            'hidden_size': 2048,
            'num_layers': 24,
            'num_attention_heads': 16,
            'ffn_hidden_size': None,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 1024,
            'type_vocab_size': 16,
            'initializer_range': 0.02,
            'use_recompute': True,
            'recompute_granularity': 'full',
            'fused_linear': True,
        }

        self.Distributed = {
            'dp_degree': 1,
            'mp_degree': 1,
            'pp_degree': 1,
            'sharding': {
                'sharding_degree': 1,
                'sharding_stage': 1,
                'sharding_offload': False,
            },
        }

        self.Fused = {'tensor_fusion': False, }

        self.Optimizer = {
            'weight_decay': 0.0,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1.0e-8,
            'lr': {
                'decay_steps': 360000,
                'warmup_rate': 0.01,
                'max_lr': 1.0e-5,
                'min_lr': 5.0e-5,
            },
            'grad_clip': 0.0,
        }

        self._update(yaml_dict)

    def _update(self, yaml_dict):
        for k in self.keys():
            self._traverse(self, k, yaml_dict)

    def _traverse(self, ori_dict, k, yaml_dict):
        if k in yaml_dict.keys():
            for ik in yaml_dict[k].keys():
                if isinstance(ori_dict[k][ik], dict):
                    self._traverse(ori_dict[k], ik, yaml_dict[k])
                else:
                    if ori_dict[k][ik] is not None and yaml_dict[k][
                            ik] is None:
                        pass
                    else:
                        ori_dict[k][ik] = yaml_dict[k][ik]

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, content):
        return copy.deepcopy(dict(self))


class GPTAutoConfig(GPTConfig):
    def __init__(self, yaml_dict, **kw):

        self.Global = {
            'device': 'gpu',
            'seed': 1024,
        }

        self.Engine = {
            'max_steps': 500000,
            'num_train_epochs': 1,
            'accumulate_steps': 1,
            'logging_freq': 1,
            'eval_freq': 500,
            'eval_iters': 10,
            'test_iters': None,
            'mix_precision': {
                'use_pure_fp16': True,
                'scale_loss': 32768.0,
                'custom_black_list': [
                    "reduce_sum", "c_softmax_with_cross_entropy",
                    "elementwise_div"
                ],
                'custom_white_list': ["lookup_table", "lookup_table_v2"],
            },
            'use_recompute': True,
            'save_load': {
                'save_steps': 1000,
                'output_dir': None,
                'ckpt_dir': None,
            }
        }

        self.Data = {
            'batch_size': {
                'global_batch_size': None,
                # 'local_batch_size': None,
                # 'micro_batch_size': 8,
            },
            'dataset': {
                'input_dir': None,
                'split': '949,50,1',
                'max_seq_len': 1024,
            }
        }

        self.Model = {
            'vocab_size': 50304,
            'hidden_size': 2048,
            'num_layers': 24,
            'num_attention_heads': 16,
            'ffn_hidden_size': None,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 1024,
            'type_vocab_size': 16,
            'initializer_range': 0.02,
        }

        self.Distributed = {
            'dp_degree': 1,
            'mp_degree': 1,
            'pp_degree': 1,
            'sharding': {
                'sharding_degree': 1,
                'sharding_stage': 1,
            },
        }

        self.Optimizer = {
            'weight_decay': 0.0,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1.0e-8,
            'lr': {
                'decay_steps': 360000,
                'warmup_rate': 0.01,
                'max_lr': 1.0e-5,
                'min_lr': 5.0e-5,
            },
            'grad_clip': 0.0,
        }

        self._update(yaml_dict)
