import os
import sys
import time
import argparse
import functools
import numpy as np
import paddle
import paddle.fluid as fluid
import resnet
import sklearn
import reader
from verification import evaluate
from utility import add_arguments, print_arguments
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.transpiler.details.program_utils import program_to_code
from paddle.fluid.optimizer import Optimizer

parser = argparse.ArgumentParser(description="Train parallel face network.")
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('train_batch_size', int,   128,         "Minibatch size for training.")
add_arg('test_batch_size',  int,   120,         "Minibatch size for test.")
add_arg('num_epochs',       int,   120,         "Number of epochs to run.")
add_arg('image_shape',      str,   "3,112,112", "Image size in the format of CHW.")
add_arg('emb_dim',          int,   512,         "Embedding dim size.")
add_arg('class_dim',        int,   85742,       "Number of classes.")
add_arg('model_save_dir',   str,   None,        "Directory to save model.")
add_arg('pretrained_model', str,   None,        "Directory for pretrained model.")
add_arg('lr',               float, 0.1,         "Initial learning rate.")
add_arg('model',            str,   "ResNet_ARCFACE50", "The network to use.")
add_arg('loss_type',        str,   "softmax",   "Type of network loss to use.")
add_arg('margin',           float, 0.5,         "Parameter of margin for arcface or dist_arcface.")
add_arg('scale',            float, 64.0,        "Parameter of scale for arcface or dist_arcface.")
add_arg('with_test',        bool, False,        "Whether to do test during training.")
# yapf: enable
args = parser.parse_args()


model_list = [m for m in dir(resnet) if "__" not in m]


def optimizer_setting(params, args):
    ls = params["learning_strategy"]
    step = 1
    bd = [step * e for e in ls["epochs"]]
    base_lr = params["lr"]
    lr = [base_lr * (0.1 ** i) for i in range(len(bd) + 1)]
    print("bd: {}".format(bd))
    print("lr_step: ".format(lr))
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(5e-4))

    if args.loss_type in ["dist_softmax", "dist_arcface"]:
        wrapper = DistributedClassificationOptimizer(
            optimizer, args.train_batch_size)
    elif args.loss_type in ["softmax", "arcface"]:
        wrapper = optimizer
    return wrapper


def train(args):
    model_name = args.model
    assert model_name in model_list, \
        "{} is not in supported lists: {}".format(args.model, model_list)
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir

    image_shape = [int(m) for m in args.image_shape.split(",")]

    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
    worker_num = int(os.getenv("PADDLE_TRAINERS_NUM", 1))

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = resnet.__dict__[model_name]()
    emb, loss = model.net(input=image,
                          label=label,
                          emb_dim=args.emb_dim,
                          class_dim=args.class_dim,
                          loss_type=args.loss_type,
                          margin=args.margin,
                          scale=args.scale)
    if args.loss_type in ["dist_softmax", "dist_arcface"]:
        shard_prob = loss._get_info("shard_prob")
        prob_all = fluid.layers.collective._c_allgather(shard_prob,
            nranks=worker_num, use_calc_stream=True)
        prob_list = fluid.layers.split(prob_all, dim=0,
            num_or_sections=worker_num)
        prob = fluid.layers.concat(prob_list, axis=1)
        label_all = fluid.layers.collective._c_allgather(label,
            nranks=worker_num, use_calc_stream=True)
        acc1 = fluid.layers.accuracy(input=prob, label=label_all, k=1)
        acc5 = fluid.layers.accuracy(input=prob, label=label_all, k=5)
    elif args.loss_type in ["softmax", "arcface"]:
        prob = loss[1]
        loss = loss[0]
        acc1 = fluid.layers.accuracy(input=prob, label=label, k=1)
        acc5 = fluid.layers.accuracy(input=prob, label=label, k=5)

    startup_prog = fluid.default_startup_program()
    train_prog = fluid.default_main_program()
    test_program = train_prog.clone(for_test=True)

    # parameters from model and arguments
    params = model.params
    params["lr"] = args.lr
    params["num_epochs"] = args.num_epochs
    params["learning_strategy"]["batch_size"] = args.train_batch_size

    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    strategy = DistributedStrategy()
    strategy.mode = "collective"
    strategy.collective_mode = "grad_allreduce"

    # initialize optimizer
    optimizer = optimizer_setting(params, args)
    dist_optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
    dist_optimizer.minimize(loss)

    if args.loss_type in ["dist_softmax", "dist_arcface"]:
        global_lr = optimizer._optimizer._global_learning_rate()
    elif args.loss_type in ["softmax", "arcface"]:
        global_lr = optimizer._global_learning_rate()

    origin_prog = fleet._origin_program
    train_prog = fleet.main_program
    if trainer_id == 0:
        with open('start.program', 'w') as fout:
            program_to_code(startup_prog, fout, True)
        with open('main.program', 'w') as fout:
            program_to_code(train_prog, fout, True)
        with open('origin.program', 'w') as fout:
            program_to_code(origin_prog, fout, True)

    gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
    place = fluid.CUDAPlace(gpu_id)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if pretrained_model:
        pretrained_model = os.path.join(pretrained_model, str(trainer_id))
        def if_exist(var):
            has_var = os.path.exists(os.path.join(pretrained_model, var.name))
            if has_var:
                print('var: %s found' % (var.name))
            return has_var
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_reader = paddle.batch(reader.arc_train(args.class_dim),
        batch_size=args.train_batch_size)
    if args.with_test:
        test_list, test_name_list = reader.test()
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    fetch_list_train = [loss.name, global_lr.name, acc1.name, acc5.name]
    fetch_list_test = [emb.name, acc1.name, acc5.name]
    # test_program = test_program._prune(targets=loss)

    num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
    real_batch_size = args.train_batch_size * num_trainers
    local_time = 0.0
    nsamples = 0
    inspect_steps = 30
    step_cnt = 0
    for pass_id in range(params["num_epochs"]):
        train_info = [[], [], [], []]
        local_train_info = [[], [], [], []]
        for batch_id, data in enumerate(train_reader()):
            nsamples += real_batch_size
            t1 = time.time()
            loss, lr, acc1, acc5 = exe.run(train_prog, feed=feeder.feed(data),
                fetch_list=fetch_list_train, use_program_cache=True)
            t2 = time.time()
            period = t2 - t1
            local_time += period
            train_info[0].append(np.array(loss)[0])
            train_info[1].append(np.array(lr)[0])
            local_train_info[0].append(np.array(loss)[0])
            local_train_info[1].append(np.array(lr)[0])
            if batch_id % inspect_steps == 0:
                avg_loss = np.mean(local_train_info[0])
                avg_lr = np.mean(local_train_info[1])
                print("Pass:%d batch:%d lr:%f loss:%f qps:%.2f acc1:%.4f acc5:%.4f" % (
                    pass_id, batch_id, avg_lr, avg_loss, nsamples / local_time,
                    acc1, acc5))
                local_time = 0
                nsamples = 0
                local_train_info = [[], [], [], []]
            step_cnt += 1

            if args.with_test and step_cnt % 2000 == 0:
                for i in xrange(len(test_list)):
                    data_list, issame_list = test_list[i]
                    embeddings_list = []
                    for j in xrange(len(data_list)):
                        data = data_list[j]
                        embeddings = None
                        beg = 0
                        while beg < data.shape[0]:
                            end = min(beg + args.test_batch_size, data.shape[0])
                            count = end - beg
                            _data = []
                            for k in xrange(end - args.test_batch_size, end):
                                _data.append((data[k], 0))
                            [_embeddings, acc1, acc5] = exe.run(test_program, 
                                fetch_list = fetch_list_test, feed=feeder.feed(_data),
                                use_program_cache=True)
                            if embeddings is None:
                                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
                            embeddings[beg:end, :] = _embeddings[(args.test_batch_size-count):, :]
                            beg = end
                        embeddings_list.append(embeddings)

                    xnorm = 0.0
                    xnorm_cnt = 0
                    for embed in embeddings_list:
                        xnorm += np.sqrt((embed * embed).sum(axis=1)).sum(axis=0)
                        xnorm_cnt += embed.shape[0]
                    xnorm /= xnorm_cnt

                    embeddings = embeddings_list[0] + embeddings_list[1]
                    #embeddings /= np.sqrt((embeddings * embeddings).sum(axis=1))[:, None]
                    embeddings = sklearn.preprocessing.normalize(embeddings)
                    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
                    acc, std = np.mean(accuracy), np.std(accuracy)

                    print('[%s][%d]XNorm: %f' % (test_name_list[i], step_cnt, xnorm))
                    print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (test_name_list[i], step_cnt, acc, std))
                    sys.stdout.flush()

        train_loss = np.array(train_info[0]).mean()
        print("End pass {0}, train_loss {1}".format(pass_id, train_loss))
        sys.stdout.flush()

        #save model
        #if trainer_id == 0:
        model_path = os.path.join(model_save_dir + '/' + model_name,
                              str(pass_id), str(trainer_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)
        
def main():
    global args
    all_loss_types = ["softmax", "arcface", "dist_softmax", "dist_arcface"]
    assert args.loss_type in all_loss_types, \
        "All supported loss types [{}], but give {}.".format(
            all_loss_types, args.loss_type)
    print_arguments(args)
    train(args)


class DistributedClassificationOptimizer(Optimizer):
    '''
    A optimizer wrapper to generate backward network for distributed
    classification training of model parallelism.
    '''

    def __init__(self, optimizer, batch_size):
        self._optimizer = optimizer
        self._batch_size = batch_size

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        assert loss._get_info('shard_logit')

        shard_logit = loss._get_info('shard_logit')
        shard_prob = loss._get_info('shard_prob')
        shard_label = loss._get_info('shard_label')
        shard_dim = loss._get_info('shard_dim')

        op_maker = fluid.core.op_proto_and_checker_maker
        op_role_key = op_maker.kOpRoleAttrName()
        op_role_var_key = op_maker.kOpRoleVarAttrName()
        backward_role = int(op_maker.OpRole.Backward)
        loss_backward_role = int(op_maker.OpRole.Loss) | int(
            op_maker.OpRole.Backward)

        # minimize a scalar of reduce_sum to generate the backward network
        scalar = fluid.layers.reduce_sum(shard_logit)
        ret = self._optimizer.minimize(scalar)

        block = loss.block
        # remove the unnecessary ops
        index = 0
        for i, op in enumerate(block.ops):
            if op.all_attrs()[op_role_key] == loss_backward_role:
                index = i
                break

        assert block.ops[index - 1].type == 'reduce_sum'
        assert block.ops[index].type == 'fill_constant'
        assert block.ops[index + 1].type == 'reduce_sum_grad'
        block._remove_op(index + 1)
        block._remove_op(index)
        block._remove_op(index - 1)

        # insert the calculated gradient 
        dtype = shard_logit.dtype
        shard_one_hot = fluid.layers.create_tensor(dtype, name='shard_one_hot')
        block._insert_op(
            index - 1,
            type='one_hot',
            inputs={'X': shard_label},
            outputs={'Out': shard_one_hot},
            attrs={
                'depth': shard_dim,
                'allow_out_of_range': True,
                op_role_key: backward_role
            })
        shard_logit_grad = fluid.layers.create_tensor(
            dtype, name=fluid.backward._append_grad_suffix_(shard_logit.name))
        block._insert_op(
            index,
            type='elementwise_sub',
            inputs={'X': shard_prob,
                    'Y': shard_one_hot},
            outputs={'Out': shard_logit_grad},
            attrs={op_role_key: backward_role})
        block._insert_op(
            index + 1,
            type='scale',
            inputs={'X': shard_logit_grad},
            outputs={'Out': shard_logit_grad},
            attrs={
                'scale': 1.0 / self._batch_size,
                op_role_key: loss_backward_role
            })
        return ret


if __name__ == '__main__':
    main()
