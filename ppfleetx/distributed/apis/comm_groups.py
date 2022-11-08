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

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.base.strategy_group import (
    StrategyGroupBase,
    DPGroup,
    MPGroup,
    PPGroup,
    ShardingGroup, )
from paddle.distributed.fleet.base.orthogonal_strategy import OrthogonalStrategy


def create_hcg(strategy, hcg_name):
    if hcg_name == "HybridCommunicateGroup":
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
    else:
        dist.init_parallel_env()
        hcg = eval("{}".format(hcg_name))(strategy)

    return hcg


class MoEGroup(StrategyGroupBase):
    """
    The communication group strategy for expert parallel.
    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.
    Returns:
        The instance of expert parallel strategy group.
    """

    def __init__(self, list_of_ranks):
        super(MoEGroup, self).__init__(list_of_ranks)
        assert not isinstance(
            self.group,
            list), "Rank {} belongs to multi moe groups".format(self._rank)


class Hybrid4DCommGroup(OrthogonalStrategy):
    def __init__(self, list_of_strategy=None, fused_strategy_dict={}):
        list_of_strategy = [
            ("dp", 1, DPGroup),
            ("mp", 1, MPGroup),
            ("pp", 1, PPGroup),
            ("sharding", 1, ShardingGroup),
        ] if list_of_strategy is None else list_of_strategy

        fused_strategy_dict["check"] = ["mp", "pp"]

        super().__init__(list_of_strategy, fused_strategy_dict)

    # data parallel
    def get_data_parallel_rank(self):
        return self.rank_in_strategy("dp")

    def get_data_parallel_world_size(self):
        return self.strategy_group("dp").world_size

    def get_data_parallel_group(self):
        return self.strategy_group("dp").group

    def get_data_parallel_group_src_rank(self):
        return self.strategy_group("dp").group.ranks[0]

    # tensor parallel
    def get_model_parallel_rank(self):
        return self.rank_in_strategy("mp")

    def get_model_parallel_world_size(self):
        return self.strategy_group("mp").world_size

    def get_model_parallel_group(self):
        return self.strategy_group("mp").group

    def get_model_parallel_group_src_rank(self):
        return self.strategy_group("mp").group.ranks[0]

    # pipeline parallel
    def get_stage_id(self):
        return self.rank_in_strategy("pp")

    def get_pipe_parallel_world_size(self):
        return self.strategy_group("pp").world_size

    def get_pipe_parallel_group(self):
        return self.strategy_group("pp").group

    def get_p2p_groups(self):
        return (self.strategy_group("pp").p2p_groups)

    # group sharded parallel
    def get_sharding_parallel_rank(self):
        return self.rank_in_strategy("sharding")

    def get_sharding_parallel_world_size(self):
        return self.strategy_group("sharding").world_size

    def get_sharding_parallel_group(self):
        return self.strategy_group("sharding")

    def get_sharding_parallel_group_src_rank(self):
        return self.strategy_group("sharding").ranks[0]

    # check parallel group
    def get_check_parallel_group(self):
        return self.strategy_group("check").group


class HybridCommGroupForMoE(Hybrid4DCommGroup):
    def __init__(self, strategy):
        self._dp_degree = strategy.hybrid_configs.get("dp_degree", 1)
        self._mp_degree = strategy.hybrid_configs.get("mp_degree", 1)
        self._pp_degree = strategy.hybrid_configs.get("pp_degree", 1)
        self._sharding_degree = strategy.hybrid_configs.get("sharding_degree",
                                                            1)

        assert self._pp_degree == 1, "The strategy combination of moe and pp \
            has not been supported in ppfleetx right now."

        assert self._sharding_degree == 1, "The strategy combination of moe and sharding \
            has not been supported in ppfleetx right now."

        list_of_strategy = [
            ("dp", self._dp_degree, DPGroup),
            ("mp", self._mp_degree, MPGroup),
            ("pp", self._pp_degree, PPGroup),
            ("sharding", self._sharding_degree, ShardingGroup),
        ]
        fused_strategy_dict = {"moe": ["dp", "mp"]}

        super().__init__(list_of_strategy, fused_strategy_dict)

    def get_expert_parallel_world_size(self):
        return self.fused_strategy_group("moe").world_size

    def get_expert_parallel_group(self):
        return self.fused_strategy_group("moe").group
