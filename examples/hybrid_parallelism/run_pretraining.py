# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import os
import sys
import time
import glob
import random

os.environ['FLAGS_enable_parallel_graph'] = "0"
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.1"
os.environ['FLAGS_sync_nccl_allreduce'] = "1"
os.environ['FLAGS_eager_delete_tensor_gb'] = "0"
os.environ['FLAGS_fuse_parameter_memory_size'] = "32"
os.environ['FLAGS_fuse_parameter_groups_size'] = "50"
os.environ['FLAGS_check_nan_inf'] = "0"

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import paddle.fluid.profiler as profiler
from paddle.distributed.fleet.meta_optimizers.sharding.utils import add_sync_comm, save_persistables

from pretraining_args import define_args
from utils.init import init_checkpoint, init_pretraining_params
from utils.topo import Topology
from utils.random import get_rng_state_tracker
from propeller import log
from model.ernie import ErnieModel, ErnieConfig

if paddle.is_compiled_with_cuda():
    device = 'gpu'
    ascend = False
    int_type = 'int64'
    try:
        from tensorboardX import SummaryWriter as LogWriter
        log.info('using tensorboard')
    except ImportError:
        from visualdl import LogWriter
        log.info('tensorboard not found, using visualdl')
elif paddle.is_compiled_with_npu():
    device = 'npu'
    ascend = True
    int_type = 'int32'
else:
    raise Exception('paddle must compiled with cuda or npu')

paddle.enable_static()
fleet.init(is_collective=True)
np.set_printoptions(threshold=1e6)

def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ Applies linear warmup of learning rate from 0 and decay to 0."""
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=learning_rate,
                    decay_steps=num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)
        return lr


def create_model(args, phase, micro_bsz, dp_sharding_rank, dp_sharding_worldsize, topo, acc_steps):
    from reader.pretraining_ds_mlm import make_pretrain_dataset

    input_fields = {
        'names': ['src_ids', 'sent_ids', 'pos_ids', 'mask_label', 'mask_pos'],
        'shapes': [[-1, args.max_seq_len], [-1, args.max_seq_len], [-1, args.max_seq_len], [-1, 1], [-1, 1]],
        'dtypes': [int_type, int_type, int_type, int_type, int_type],
        'lod_levels': [0, 0, 0, 0, 0],
    }

    with fluid.device_guard(f"{device}:0"):
        inputs = [fluid.data(name=input_fields['names'][i],
                      shape=input_fields['shapes'][i],
                      dtype=input_fields['dtypes'][i],
                      lod_level=input_fields['lod_levels'][i]) for i in range(len(input_fields['names']))]
    (src_ids, sent_ids, pos_ids, mask_label, mask_pos) = inputs
    train_file_list = glob.glob(args.data_dir + "/*")
    vocab = {}
    with open(args.vocab_file) as r:
        for line in r:
            lines = line.strip().split('\t')
            vocab[lines[0]] = int(lines[1])

    log.debug("========= dp_sharding worker: {} of {} ==========".format(dp_sharding_rank, dp_sharding_worldsize))

    data_reader = make_pretrain_dataset('pt', train_file_list, True, vocab, micro_bsz, len(vocab),
            args.max_seq_len, dp_sharding_rank, dp_sharding_worldsize, device)
    with fluid.device_guard(f"{device}:0"):
        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=inputs, capacity=70, iterable=False)

    if not ascend:
        places = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    else:
        places = fluid.NPUPlace(int(os.environ.get('FLAGS_selected_npus', 0)))

    def data_gen():
        yield from data_reader

    data_loader.set_batch_generator(data_gen, places)

    ernie_config = ErnieConfig(args.ernie_config_file)._config_dict
    ernie_config["preln"] = args.preln

    # NOTE when trying to compare the loss between mp with non-parallelism-sinlge model, 
    # should disable weight share for non-parallelism-sinlge run
    weight_sharing = False

    with fluid.device_guard(f"{device}:0"):
        ernie = ErnieModel(src_ids,
                           sent_ids,
                           pos_ids,
                           ernie_config,
                           weight_sharing=weight_sharing,
                           topo=topo,
                           device=device)
    checkpoints = ernie._checkpoints
    checkpoints.pop(-1)

    with fluid.device_guard(f'{device}:{args.num_pp-1}'):
        mask_lm_loss, mean_mask_lm_loss = ernie.get_lm_output(mask_label, mask_pos)
        total_loss = mean_mask_lm_loss

        pp_total_loss = None
        if topo.pp.size > 1:
            mask_lm_loss.persistable = True
            mean_mask_lm_loss.persistable = True
            total_loss.persistable = True

            # To compute the total loss across micro batches
            pp_total_loss = paddle.fluid.layers.fill_constant([1,], 'float32', 0.0)
            pp_total_loss.persistable = True
            block = fluid.default_main_program().global_block()
            tmp = total_loss / acc_steps
            block.append_op(
                type='elementwise_add',
                inputs={'X': [pp_total_loss],
                        'Y': [tmp]},
                outputs={'Out': [pp_total_loss]}
            )

    graph_vars = {
        'data_loader': data_loader,
        'mask_lm_loss': mask_lm_loss,
        'mean_mask_lm_loss': mean_mask_lm_loss,
        'total_loss': total_loss,
        'pp_total_loss': pp_total_loss,
        'checkpoints': checkpoints,
    }
    return graph_vars

def create_broadcast_program(ref_program, ring_id = 0, root_rank = 0):
    broadcast_program = fluid.Program()
    from paddle.fluid.framework import Parameter
    def is_parameter(var):
        return isinstance(var, Parameter)
    parameter_list = list(filter(is_parameter, ref_program.list_vars()))

    # create vars
    broadcast_main_block = broadcast_program.global_block()
    broadcast_params = []
    for param in parameter_list:
        if not param.is_distributed:
            print("[broadcast program] create var: {}".format(param.name))
            gradient_merge_var = broadcast_main_block.create_parameter(
                name=param.name,
                shape=param.shape,
                dtype=param.dtype,
                type=param.type,
                stop_gradient=param.stop_gradient)
            broadcast_params.append(param)

    # append ops
    with fluid.program_guard(broadcast_program, None):
        with fluid.unique_name.guard():
            for param in broadcast_params:
                fluid.layers.collective._c_broadcast(param, root=root_rank, ring_id=ring_id, use_calc_stream=True)
        
    return broadcast_program

def train(args):
    print("to run startup")
    log.info("pretraining start")

    if not ascend:
        place = fluid.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))
    else:
        place = fluid.NPUPlace(int(os.environ.get('FLAGS_selected_npus', 0)))

    # set seed
    if args.debug:
        random.seed(args.seed)
        np.random.seed(args.seed)
    paddle.seed(args.seed)
    get_rng_state_tracker().add('global_seed', args.seed)
    get_rng_state_tracker().add('local_seed', args.seed + fleet.worker_index() + 2021)

    # define distribution strategy
    dist_strategy = fleet.DistributedStrategy()
    if args.use_recompute:
       log.info("using recompute.")
    dist_strategy.recompute = args.use_recompute
    dist_strategy.sharding = args.use_sharding
    dist_strategy.pipeline = args.num_pp > 1

    # define topology structure for dp/pp/mp
    topo = Topology(rank=fleet.worker_index(),
                    world_size=fleet.worker_num(),
                    dp=args.num_dp,
                    pp=args.num_pp,
                    sharding=args.num_sharding,
                    mp=args.num_mp)

    is_last = False
    if topo.pp.rank == (topo.pp.size -1):
        is_last = True

    dp_sharding_rank = topo.dp.rank * topo.sharding.size + topo.sharding.rank
    # dp and sharding are both data parallelism and need to take an unique partition of global batch size
    dp_sharding_worldsize = topo.dp.size * topo.sharding.size
    bsz_per_dp = args.global_bsz // dp_sharding_worldsize

    micro_bsz = args.micro_bsz
    assert args.global_bsz >= micro_bsz * dp_sharding_worldsize, f"[micro bsz x dp_sharding_worldsize] larger than global bsz, globa_bsz: {args.global_bsz} micro_bsz: {micro_bsz}, dp_sharding_worldsize: {dp_sharding_worldsize}"
    assert args.global_bsz % micro_bsz == 0, f"cannot do gradient accumulate, globa_bsz: {args.global_bsz} micro_bsz: {micro_bsz}"
    acc_steps = bsz_per_dp // micro_bsz

    if args.num_mp == 1 and args.num_pp == 1 and args.num_sharding == 1:
        # single 
        assert args.num_dp == 1, "normal data parallelism should not use sharding config"
    else:
        # sharding \ model parallel \ pipeline
        assert dist_strategy.sharding == True
        dist_strategy.sharding_configs = {"segment_broadcast_MB": 32,
                                          "sharding_degree": args.num_sharding,
                                          "mp_degree": args.num_mp,
                                          "pp_degree": args.num_pp,
                                          "dp_degree":args.num_dp,
                                          "gradient_merge_acc_step": acc_steps,
                                          "optimize_offload": False,
                                         }
        dist_strategy.pipeline_configs = {"schedule_mode": "1F1B",
                                          "micro_batch_size": micro_bsz,
                                          "accumulate_steps": acc_steps,
                                         }
    log.info(f"using globa_bsz: {args.global_bsz} micro_bsz: {micro_bsz}, acc_steps: {acc_steps}")

    dist_strategy.amp = args.use_amp
    dist_strategy.amp_configs = {
        "custom_white_list": ['softmax', 'layer_norm', 'gelu'],
        "init_loss_scaling": 32768,
        "decr_every_n_nan_or_inf": 2,
        "incr_every_n_steps": 1000,
        "incr_ratio": 2.0,
        "use_dynamic_loss_scaling": True,
        "use_fp16_guard": False,
        #"decr_ratio": 0.5,
        #"use_pure_fp16": False,
    }

    dist_strategy.lamb = args.use_lamb
    dist_strategy.lamb_configs = {
        'lamb_weight_decay': 0.01,
        'exclude_from_weight_decay': ['layer_norm_bias', 'layer_norm_scale', '.b_0']
    }

    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        with fluid.unique_name.guard():
            graph_vars = create_model(args, 'train', micro_bsz, dp_sharding_rank, dp_sharding_worldsize, topo, acc_steps)
            data_loader = graph_vars['data_loader']
            for op in train_program.global_block().ops:
                if op.type == 'fill_constant':
                    op._set_attr('op_device', f"{device}:0") # XXX: hack: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/layers/tensor.py#L1376

            if args.use_recompute:
                dist_strategy.recompute_configs = {
                    "checkpoints": graph_vars['checkpoints'],
                }
                                                  
            log.debug("base lr: {}".format(args.learning_rate))

            warmup_steps = args.warmup_steps
            num_train_steps = args.num_train_steps
            # refine the lr decay for sharding-gm
            if args.num_sharding > 1 and args.num_pp == 1 and acc_steps > 1:
                warmup_steps = warmup_steps * acc_steps
                num_train_steps = num_train_steps * acc_steps
            scheduled_lr = linear_warmup_decay(learning_rate=args.learning_rate,
                                               warmup_steps=warmup_steps,
                                               num_train_steps=num_train_steps)

            clip_norm_thres = 1.0
            if args.num_pp > 1:
                pp_loss = graph_vars['pp_total_loss']
                pp_loss_name = pp_loss.name
                for op in train_program.global_block().ops:
                    if op.type == "fill_constant" and op.desc.output_arg_names()[0] == pp_loss_name:
                        op._set_attr("op_role", 16)
                        op._set_attr('op_device', f"{device}:%d"%(args.num_pp-1))
            optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr,
                                             grad_clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm_thres),
                                            )
            optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
            optimizer.minimize(graph_vars['total_loss'])

            final_strategy = fleet._final_strategy()
            applied_meta_list = fleet._get_applied_meta_list()
            if device == "gpu":
                log.info(f"using dist strategy: {dist_strategy}") 
                log.info("final strategy: {}".format(final_strategy))
                log.info("applied_meta_list: {}".format(applied_meta_list))

    if args.num_mp > 1 and args.num_pp == 1:
        broadcast_program = create_broadcast_program(train_program)
    elif args.num_mp > 1 and args.num_pp > 1:
        broadcast_program = create_broadcast_program(train_program._pipeline_opt['section_program'])

    # save program
    program_desc_dir = os.path.join(args.output_dir, "program_desc")
    if not os.path.isdir(program_desc_dir):
        os.makedirs(program_desc_dir)

    # save the true pp program
    if args.num_pp > 1:
        with open(program_desc_dir + "/main_program.txt.%d" % (int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
            f.write(str(train_program._pipeline_opt['section_program']))
        with open(program_desc_dir + "/startup_program.txt.%d" % (int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
            f.write(str(startup_program._pipeline_opt['startup_program']))
    else:
        with open(program_desc_dir + "/main_program.txt.%d" % (int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
            f.write(str(train_program))
        with open(program_desc_dir + "/startup_program.txt.%d" % (int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
            f.write(str(startup_program))

    if args.num_mp > 1:
        with open(program_desc_dir + "/broadcast_program.txt.%d" % (int(os.environ.get('FLAGS_selected_gpus', 0))), 'w') as f:
            f.write(str(broadcast_program))

    exe = fluid.Executor(place)
    # # TODO
    if args.num_mp > 1 :
        paddle.seed(2021 + int(os.environ.get(f'FLAGS_selected_{device}s', 0)))
    exe.run(startup_program)

    if args.num_pp == 1 and args.num_mp == 1 and args.num_dp == 1:
        paddle.fluid.io.save_persistables(exe, './saved_model', main_program=train_program)
    elif args.debug:
        def predicate(var):
            if os.path.exists('./saved_model/%s'%var.name):
                return True
            return False
        if args.num_pp > 1:
            paddle.fluid.io.load_vars(exe, "./saved_model", main_program=train_program._pipeline_opt['section_program'], predicate=predicate)
        else:
            paddle.fluid.io.load_vars(exe, "./saved_model", main_program=train_program, predicate=predicate)

    if args.use_amp:
        optimizer.amp_init(place)

    output_dir = args.output_dir + '_' + str(fleet.worker_index())
    save_steps = args.save_steps
    total_time = 0
    cost_vals, lm_losses, sop_accs = [], [], []
    global_steps = args.global_steps + 1
    steps = 0
    log_path = 'train_log/node-%d' % fleet.worker_index()
    start_time = time.time()

    # # TODO
    if args.num_mp > 1:
        exe.run(broadcast_program, fetch_list=[])
        print("Done broadcast")

    if device == "gpu":
        with LogWriter(os.path.join(args.output_dir, log_path)) as swriter:
            data_loader.start()
            while True:
                if not is_last:
                    fetch_list=[]
                else:
                    fetch_list = [graph_vars['total_loss'],
                                graph_vars['mean_mask_lm_loss'],
                                scheduled_lr]
                    if args.use_amp:
                        loss_scaling = train_program.global_block().vars['loss_scaling_0']
                        fetch_list.append(loss_scaling)

                    if args.num_pp > 1:
                        pp_total_loss = graph_vars['pp_total_loss']
                        fetch_list.append(pp_total_loss)
                
                log.info(f"********exe.run_{steps}******* ")        
                ret = exe.run(train_program, fetch_list=fetch_list, use_program_cache=True) # run one mini-batch(=acc_steps micro-batch)
                log.info("*******exe .run .end ******")
                steps += 1

                if is_last:
                    if args.use_amp and args.num_pp <= 1:
                        cost_val, lm_loss, lr, loss_scaling_0 = ret
                    elif args.num_pp > 1 and args.use_amp:
                        cost_val, lm_loss, lr, loss_scaling_0, pp_loss = ret
                    elif args.num_pp > 1:
                        cost_val, lm_loss, lr, pp_loss = ret
                    else:
                        cost_val, lm_loss, lr = ret
                    cost_vals.append(cost_val[0])
                    lm_losses.append(lm_loss[0])

                    if steps > 0 and (steps % args.log_steps)  == 0:
                        end_time = time.time()
                        total_time = end_time - start_time
                        cost_val = np.mean(cost_vals)
                        lm_loss = np.mean(lm_losses)
                        swriter.add_scalar('loss/total_loss', cost_val, steps)
                        swriter.add_scalar('loss/mlm_loss', lm_loss, steps)
                        swriter.add_scalar('lr/scheduled_lr', lr[0], steps)

                        if args.use_amp:
                            swriter.add_scalar('lr/loss_scaling', loss_scaling_0[0], steps)
                        else:
                            loss_scaling_0 = [0.0]
                        if args.num_pp <= 1:
                            pp_loss = [0.0]

                        log.info(
                            "worker_index: %d, step: %d, cost: %f, "
                            "mlm loss: %f, "
                            "speed: %f steps/s, "
                            "speed: %f samples/s, "
                            "speed: %f tokens/s, "
                            "learning rate: %.3e, loss_scalings: %f, "
                            "pp_loss: %f"
                            % (fleet.worker_index(), steps, cost_val,
                            lm_loss,
                            args.log_steps / total_time,
                            args.log_steps * args.global_bsz / total_time,
                            args.log_steps * args.global_bsz * args.max_seq_len / total_time,
                            lr[0], loss_scaling_0[0], pp_loss[0]))

                        cost_vals, lm_losses, sop_accs = [], [], []
                        start_time = time.time()

                if steps > 0 and args.save_steps > 0 and steps % args.save_steps == 0:
                    if args.use_hybrid_dp and fleet.worker_index() > 8:
                        continue
                    save_path = os.path.join(output_dir, 'step_' + str(steps))
                    log.debug("saving models to {}".format(save_path))
                    save_persistables(exe, save_path, train_program)

                if steps == num_train_steps:
                    if args.use_hybrid_dp and fleet.worker_index() > 8:
                        continue
                    save_path = os.path.join(output_dir, 'final_step_' + str(steps))
                    save_persistables(exe, save_path, train_program)
                    log.debug("saving final models to {}".format(save_path))
                    log.debug("end of training, total steps: {}".format(steps))
                    break
            data_loader.reset()

    else:
        data_loader.start()
        while True:
            if not is_last:
                fetch_list=[]
            else:
                fetch_list = [graph_vars['total_loss'],
                              graph_vars['mean_mask_lm_loss'],
                              scheduled_lr]
                if args.use_amp:
                    loss_scaling = train_program.global_block().vars['loss_scaling_0']
                    fetch_list.append(loss_scaling)

                if args.num_pp > 1:
                    pp_total_loss = graph_vars['pp_total_loss']
                    fetch_list.append(pp_total_loss)

            log.info(f"********exe.run_{steps}******* ")
            ret = exe.run(train_program, fetch_list=fetch_list, use_program_cache=True) # run one mini-batch(=acc_steps micro-batch)
            log.info("*******exe .run .end ******")
            steps += 1

            if is_last:
                if args.use_amp and args.num_pp <= 1:
                    cost_val, lm_loss, lr, loss_scaling_0 = ret
                elif args.num_pp > 1 and args.use_amp:
                    cost_val, lm_loss, lr, loss_scaling_0, pp_loss = ret
                elif args.num_pp > 1:
                    cost_val, lm_loss, lr, pp_loss = ret
                else:
                    cost_val, lm_loss, lr = ret
                cost_vals.append(cost_val[0])
                lm_losses.append(lm_loss[0])

                if steps > 0 and (steps % args.log_steps)  == 0:
                    end_time = time.time()
                    total_time = end_time - start_time
                    cost_val = np.mean(cost_vals)
                    lm_loss = np.mean(lm_losses)
                    log.info(f'loss/total_loss, {cost_val}, {steps}')
                    log.info(f'loss/mlm_loss, {lm_loss}, {steps}')
                    log.info(f'lr/scheduled_lr, {lr[0]}, {steps}')

                    if args.use_amp:
                        log.info(f'lr/loss_scaling, {loss_scaling_0[0]}, {steps}')
                    else:
                        loss_scaling_0 = [0.0]
                    if args.num_pp <= 1:
                        pp_loss = [0.0]

                    log.info(
                        "worker_index: %d, step: %d, cost: %f, "
                        "mlm loss: %f, "
                        "speed: %f steps/s, "
                        "speed: %f samples/s, "
                        "speed: %f tokens/s, "
                        "learning rate: %.3e, loss_scalings: %f, "
                        "pp_loss: %f"
                        % (fleet.worker_index(), steps, cost_val,
                        lm_loss,
                        args.log_steps / total_time,
                        args.log_steps * args.global_bsz / total_time,
                        args.log_steps * args.global_bsz * args.max_seq_len / total_time,
                        lr[0], loss_scaling_0[0], pp_loss[0]))

                    cost_vals, lm_losses, sop_accs = [], [], []
                    start_time = time.time()

            if steps > 0 and args.save_steps > 0 and steps % args.save_steps == 0:
                if args.use_hybrid_dp and fleet.worker_index() > 8:
                    continue
                save_path = os.path.join(output_dir, 'step_' + str(steps))
                log.debug("saving models to {}".format(save_path))
                save_persistables(exe, save_path, train_program)

            if steps == num_train_steps:
                if args.use_hybrid_dp and fleet.worker_index() > 8:
                    continue
                save_path = os.path.join(output_dir, 'final_step_' + str(steps))
                save_persistables(exe, save_path, train_program)
                log.debug("saving final models to {}".format(save_path))
                log.debug("end of training, total steps: {}".format(steps))
                break
        data_loader.reset()


if __name__ == "__main__":
    args = define_args()
    train(args)
