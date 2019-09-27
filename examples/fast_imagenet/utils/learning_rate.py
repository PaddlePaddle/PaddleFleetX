from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
from paddle.fluid.initializer import init_on_cpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import math


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        epoch = ops.floor(global_step / step_each_epoch)
        decayed_lr = learning_rate * \
                     (ops.cos(epoch * (math.pi / epochs)) + 1)/2
    return decayed_lr


def lr_warmup(learning_rate, warmup_steps, start_lr, end_lr):
    """ Applies linear learning rate warmup for distributed training
        Argument learning_rate can be float or a Variable
        lr = lr + (warmup_rate * step / warmup_steps)
    """
    assert (isinstance(end_lr, float))
    assert (isinstance(start_lr, float))
    linear_step = end_lr - start_lr
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate_warmup")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                decayed_lr = start_lr + linear_step * (global_step /
                                                       warmup_steps)
                fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.default():
                fluid.layers.tensor.assign(learning_rate, lr)

        return lr


def lr_linear(step_bounds, lr_list):
    """ Applies linear learning rate per segment for distributed training
    """
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate_linear")
        # default_lr = fluid.layers.fill_constant(shape=[1], 
        #     value=lr_list[4][1], dtype='float32')

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < step_bounds[0]):
                start_lr = lr_list[0][0]
                end_lr = lr_list[0][1]
                linear_step = end_lr - start_lr
                warmup_steps = step_bounds[0]
                decayed_lr = start_lr + linear_step * (global_step /
                                                       warmup_steps)
                fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.case(global_step < step_bounds[1]):
                start_lr = lr_list[1][0]
                end_lr = lr_list[1][1]
                linear_step = end_lr - start_lr
                warmup_steps = step_bounds[1] - step_bounds[0]
                step = global_step - step_bounds[0]
                decayed_lr = start_lr + linear_step * (step /
                                                       warmup_steps)
                fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.case(global_step < step_bounds[2]):
                start_lr = lr_list[2][0]
                end_lr = lr_list[2][1]
                linear_step = end_lr - start_lr
                warmup_steps = step_bounds[2] - step_bounds[1]
                step = global_step - step_bounds[1]
                decayed_lr = start_lr + linear_step * (step /
                                                       warmup_steps)
                fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.case(global_step < step_bounds[3]):
                start_lr = lr_list[3][0]
                end_lr = lr_list[3][1]
                linear_step = end_lr - start_lr
                warmup_steps = step_bounds[3] - step_bounds[2]
                step = global_step - step_bounds[2]
                decayed_lr = start_lr + linear_step * (step /
                                                       warmup_steps)
                fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.case(global_step < step_bounds[4]):
                start_lr = lr_list[4][0]
                end_lr = lr_list[4][1]
                linear_step = end_lr - start_lr
                warmup_steps = step_bounds[4] - step_bounds[3]
                step = global_step - step_bounds[3]
                decayed_lr = start_lr + linear_step * (step /
                                                       warmup_steps)
                fluid.layers.tensor.assign(decayed_lr, lr)
            with switch.default():
                decayed_lr = lr_list[4][1]
                fluid.layers.tensor.assign(default_lr, lr)

        return lr
