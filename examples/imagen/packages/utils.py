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

import builtins
import os
import math
import time
import json
import datetime
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import random

import paddle
import paddle.amp as amp
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.sharding import group_sharded_parallel, save_group_sharded_model


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = paddle.to_tensor([self.count, self.total], dtype=paddle.float64)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = paddle.to_tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = paddle.to_tensor(list(self.deque), dtype=paddle.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        try:
            for meter in self.meters.values():
                meter.synchronize_between_processes()
        except:
            print("check is_dist_avail_and_initialized()")
            pass

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
            'time: {time}', 'data: {data}'
        ]
        if 'gpu' in paddle.device.get_device():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=paddle.device.cuda.max_memory_allocated() / MB))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def setup_for_distributed_each_gpu(rank):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        builtin_print('rank is: ', rank, end=' ')
        builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.get_world_size() > 1:
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        paddle.save(*args, **kwargs)


def init_distributed_mode(args):
    dist.init_parallel_env()
    args.distributed = True
    if not args.enable_multi_print:
        setup_for_distributed(is_main_process())
    else:
        setup_for_distributed_each_gpu(get_rank())


def set_hyrbid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank,
                             device):
    assert device != "cpu"

    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add('global_seed', global_seed)
    tracker.add('local_seed', local_seed)


def init_sharding(args):
    paddle.set_device(args.device)
    nranks = get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.mp_degree,
        "pp_degree": args.pp_degree,
        "sharding_degree": args.sharding_degree
    }

    strategy.pipeline_configs = {
        "accumulate_steps": args.update_freq,
        "micro_batch_size": args.batch_size
    }

    # set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}

    fleet.init(is_collective=True, strategy=strategy)

    # obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    global_rank = hcg.get_global_rank()
    mp_rank = hcg.get_model_parallel_rank()
    pp_rank = hcg.get_stage_id()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()

    # sharding stage2/3 not support hybrid parallel
    if args.sharding_stage in [2, 3]:
        assert args.dp_degree == args.mp_degree == args.pp_degree == 1, "sharding stage2/3 will support hybrid parallel later"

    sharding_size = hcg.get_sharding_parallel_world_size()
    data_world_rank = dp_rank * sharding_size + sharding_rank
    data_world_size = args.dp_degree * args.sharding_degree
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, data_world_rank, mp_rank, pp_rank,
                             args.device)


def save_model(args,
               epoch,
               model_without_ddp,
               optimizer,
               loss_scaler,
               tag=None,
               exp_name=None):
    to_save_state_dict = model_without_ddp.state_dict()
    for key in list(to_save_state_dict.keys()):
        if key.startswith('teacher_network.'):
            to_save_state_dict.pop(key)

    to_save = {
        'model': to_save_state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    if loss_scaler is not None:
        to_save['scaler'] = loss_scaler.state_dict()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_name = f'checkpoint-{tag or epoch}.pd'
    if exp_name is not None:
        checkpoint_name = f"{exp_name}_" + checkpoint_name
    save_on_master(to_save, os.path.join(args.output_dir, checkpoint_name))


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            raise NotImplementedError
        else:
            checkpoint = paddle.load(args.resume)

        checkpoint_model = checkpoint['model']
        model_without_ddp.set_state_dict(checkpoint_model)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (
                hasattr(args, 'eval') and args.eval):
            optimizer.set_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = paddle.to_tensor(x)
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0,
                     warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([
        final_value + 0.5 * (base_value - final_value) *
        (1 + math.cos(math.pi * i / (len(iters)))) for i in iters
    ])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args,
               epoch,
               model_without_ddp,
               optimizer,
               loss_scaler,
               tag=None,
               exp_name=None):
    to_save_state_dict = model_without_ddp.state_dict()

    to_save = {
        'model': to_save_state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    if loss_scaler is not None:
        to_save['scaler'] = loss_scaler.state_dict()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_name = f'checkpoint-{tag or epoch}.pd'
    if exp_name is not None:
        checkpoint_name = f"{exp_name}_" + checkpoint_name
    save_on_master(to_save, os.path.join(args.output_dir, checkpoint_name))


def auto_load_model(args, model_without_ddp, optimizer, scaler,
                    model_ema=None):
    output_dir = Path(args.output_dir)

    # amp
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(
            os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir,
                                       'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        checkpoint = paddle.load(args.resume)

        # handle ema model
        need_state_dict = model_without_ddp.state_dict()
        need_ema = False
        for key in need_state_dict.keys():
            if 'teacher_network' in key:
                need_ema = True
                break

        checkpoint_model = checkpoint['model']

        if need_ema:
            all_keys = list(checkpoint_model.keys())
            all_keys = [key for key in all_keys if key.startswith('encoder.')]
            for key in all_keys:
                new_key = key.replace('encoder.', 'teacher_network.')
                checkpoint_model[new_key] = checkpoint_model[key]

        model_without_ddp.set_state_dict(checkpoint_model)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.set_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if hasattr(args, 'model_ema') and args.model_ema:
                _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def wrap_sharding_2_3(model, optimizer, scaler, sharding_stage,
                      sharding_offload):
    group = fleet.get_hybrid_communicate_group().get_sharding_parallel_group()
    level = "p_g_os" if sharding_stage == 3 else "os_g"
    return group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level=level,
        scaler=scaler,
        group=group,
        offload=sharding_offload)
