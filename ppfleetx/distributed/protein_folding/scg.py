#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
Communication group manager
"""
import types
import numpy as np
from paddle import distributed as dist


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


class SingletonCommunicationGroup(object):
    """ A singleton communication group for hybrid parallel. """

    def __init__(self):
        self.initialized = False

    def init_process_group(self,
                           parallel_degree=[('dp', None)],
                           custom_parallel_degree=None):
        """ init the hybrid parallel process group. In most cases, only one hybrid parallel process group is 
            initialized in a distributed program, so this is a singleton design.
        
            args:
                parallel_degree(list of tuple): Each parallel strategy consists of a tuple.
                E.g. [('dp', None), ('pp', 2), ('mp', 2)], means that the data parallel degree is obtained by 
                calculation, the pipeline parallel degree is 2, and the model parallel degree is 2. For data 
                parallelism, it is special. It is assumed that data parallelism has always been in the outermost 
                dimension. If it is not set, the data parallelism degree will be automatically calculated.
                
                When multiple distributed strategies fully overlap, this can be represented by setting multiple 
                parallel names in a tuple. For example, [('dp', None), ('mp', 'bp', 2)]. Default is [('dp', None)]
                
                custom_parallel_degree(list of tuple): Higher-order usages can be used when the automatically 
                derived parallel strategy fails to meet user needs. The user can calculate the rank id in the 
                communication group and pass it in through the `custom_parallel_degree` arg. Default is None.
                E.g. [('dp', [[0, 2, 4, 6], [1, 3, 5, 7]]), ('mp', 'bp', [[0, 1], [2, 3], [4, 5], [6, 7]])]
                
            note:
                `parallel_degree` and `custom_parallel_degree` are mutually exclusive, only one can be set at 
                the same time.
                
            example 1:
                # 8 gpus on single node, dp will be 2
                # dp_group_ranks = [[0, 4], [1, 5], [2, 6], [3, 7]]
                # pp_group_ranks = [[0, 2], [1, 3], [4, 6], [5, 7]]
                # mp_group_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
                scg = SingletonCommunicationGroup()
                scg.init_process_group(parallel_degree=[('dp', None), ('pp', 2), ('mp', 2)])
                print(scg.dp_group)
                print(scg.get_rank_in_bp_group())
                print(scg.get_dp_world_size())
                
            example 2:
                # 8 gpus on single node, dp will be 2
                # dp_group_ranks = [[0, 4], [1, 5], [2, 6], [3, 7]]
                # pp_group_ranks = [[0, 2], [1, 3], [4, 6], [5, 7]]
                # mp_group_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
                scg = SingletonCommunicationGroup()
                scg.init_process_group(parallel_degree=[('pp', 2), ('mp', 2)])
                
            example 3:
                # 8 gpus on single node, dp will be 4, mp and bp share a communication group.
                # dp_group_ranks = [[0, 2, 4, 6], [1, 3, 5, 7]]
                # mp_group_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
                # bp_group_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
                scg = SingletonCommunicationGroup()
                scg.init_process_group(parallel_degree=[('dp', None), ('mp', 'bp', 2)])
                
            example 4:
                # 8 gpus on single node, dp will be 8, mp will be 8, dp and mp share a communication group.
                # dp_group_ranks = [[0, 1, 2, 3, 4, 5, 6, 7]]
                # mp_group_ranks = [[0, 1, 2, 3, 4, 5, 6, 7]]
                scg = SingletonCommunicationGroup()
                scg.init_process_group(parallel_degree=[('dp', 'mp', 8)])
                
            example 5:
                # Equal to example 3 but pass config by custom_parallel_degree.
                # dp_group_ranks = [[0, 2, 4, 6], [1, 3, 5, 7]]
                # mp_group_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
                # bp_group_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
                scg = SingletonCommunicationGroup()
                scg.init_process_group(parallel_degree=None, custom_parallel_degree=[('dp', [[0, 2, 4, 6], [1, 3, 5, 7]]), ('mp', 'bp', [[0, 1], [2, 3], [4, 5], [6, 7]])])
            
        """

        assert not (parallel_degree is not None and custom_parallel_degree is not None), \
            f"parallel_degree and custom_parallel_degree only can be set one."

        assert self.initialized == False, "Communication group is already initialized!"

        if dist.is_initialized() is not None:
            dist.init_parallel_env()

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # parse parallel_degree
        if parallel_degree is not None and custom_parallel_degree is None:

            def check_valid(inp):
                assert isinstance(
                    inp, list), f"parallel_degree must be list of tuple"
                for item in inp:
                    num_ele = len(item)
                    assert num_ele >= 2, f"each item in parallel_degree must has least two element."
                    assert isinstance(item[-1], (
                        int, type(None)
                    )), f"the last element in each item must be int or None"
                    for idx in range(num_ele - 1):
                        assert isinstance(item[idx], str)

            check_valid(parallel_degree)

            dp_exist = False
            dp_has_set = False
            num_ranks = 1
            for idx, item in enumerate(parallel_degree):
                degree = item[-1]
                if 'dp' in item:
                    assert idx == 0, 'The data parallel dimension must be the outermost dimension.'
                    dp_exist = True

                    if degree is not None:
                        dp_has_set = True
                    else:
                        degree = 1
                assert degree is not None, 'All but dp must specify the parallel degree explicitly.'
                num_ranks *= degree

            # check and update dp
            if not dp_exist:
                assert world_size % num_ranks == 0, 'The total number of parallelism products set is not divisible by the total number of cards.'
                parallel_degree.insert(0, ('dp', world_size // num_ranks))
            elif dp_exist and not dp_has_set:
                assert world_size % num_ranks == 0, 'The total number of parallelism products set is not divisible by the total number of cards.'
                parallel_degree[0] = ('dp', world_size // num_ranks)
            else:
                assert num_ranks == world_size, 'The total number of parallelism products set is not equal to the total number of cards.'

            degrees = tuple([item[-1] for item in parallel_degree])
            num_parallel = len(parallel_degree)
            group_arr = np.arange(0, world_size).reshape(degrees)

            custom_parallel_degree = []

            for idx, item in enumerate(parallel_degree):
                parallel_name = item[0]
                degree = item[-1]
                transpose_axes = []
                for axis in range(num_parallel):
                    if axis != idx:
                        transpose_axes.append(axis)
                transpose_axes.append(idx)
                arr = group_arr.transpose(transpose_axes).reshape((-1, degree))

                custom_parallel_degree.append([])

                for parallel_name in item[:-1]:
                    custom_parallel_degree[idx].append(parallel_name)
                custom_parallel_degree[idx].append([])

                for i in range(world_size // degree):
                    ranks = arr[i].tolist()
                    custom_parallel_degree[idx][-1].append(ranks)
                custom_parallel_degree[idx] = tuple(custom_parallel_degree[
                    idx])
        else:
            print(
                "We do not check the validity of user-defined custom_parallel_degree."
            )

        # new group and set attr
        for item in custom_parallel_degree:
            ranks_list = item[-1]
            for i in range(len(ranks_list)):
                ranks = ranks_list[i]
                for parallel_name in item[:-1]:
                    group = dist.new_group(ranks)
                    print(f'> {parallel_name} ranks: {ranks}')
                    if rank in ranks:
                        setattr(self, f'{parallel_name}_group', group)

                        def get_rank_in_group(parallel_name):
                            def func():
                                if not self.initialized:
                                    return -1
                                group = getattr(self, f'{parallel_name}_group')
                                return group.get_group_rank(dist.get_rank())
                            return func

                        setattr(self, f'get_rank_in_{parallel_name}_group',
                                get_rank_in_group(parallel_name))

                        def get_group_world_size(parallel_name):
                            def func():
                                if not self.initialized:
                                    return -1
                                group = getattr(self, f'{parallel_name}_group')
                                return group.nranks
                            return func

                        setattr(self, f'get_{parallel_name}_world_size',
                                get_group_world_size(parallel_name))

        self.initialized = True

scg = SingletonCommunicationGroup()
