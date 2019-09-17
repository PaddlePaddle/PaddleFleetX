import os
import sys
import time
import argparse
import functools
import numpy as np
import paddle
import paddle.fluid as fluid
import resnet
import reader
from verification import evaluate
from utility import add_arguments, print_arguments
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.transpiler.details.program_utils import program_to_code
from paddle.fluid.transpiler.collective import DistributedClassificationOptimizer

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('train_batch_size', int,   512,         "Minibatch size for training.")
add_arg('test_batch_size',  int,   120,         "Minibatch size for test.")
add_arg('num_epochs',       int,   120,         "Maximum number of epochs to run.")
add_arg('image_shape',      str,   "3,112,112", "Input image size in the format of CHW.")
add_arg('emb_dim',          int,   512,         "Embedding dim size.")
add_arg('class_dim',        int,   85184,       "Number of classes.")
add_arg('model_save_dir',   str,   "output",    "Directory to save model.")
add_arg('pretrained_model', str,   None,        "Directory for pretrained model.")
add_arg('checkpoint',       str,   None,        "Directory for checkpoint.")
add_arg('lr',               float, 0.1,         "Initial learning rate.")
add_arg('model',            str,   "ResNet_ARCFACE50", "Name of the network to use.")
add_arg('loss_type',        str,   "softmax",   "Type of network loss to use.")
add_arg('margin',           float, 0.5,         "Parameter of margin for arcface or dist_arcface.")
add_arg('scale',            float, 64.0,        "Parameter of scale for arcface or dist_arcface.")
add_arg('mode',             str,   '2',         "Margin mode to use.")
add_arg('with_test',        bool, False,        "Whether to do test during training.")
# yapf: enable
args = parser.parse_args()

model_list = [m for m in dir(resnet) if "__" not in m]

def transpile(startup_prog, train_prog):
    config = fluid.DistributeTranspilerConfig()
    config.mode = 'collective'
    config.collective_mode = 'grad_allreduce'
    t = fluid.DistributeTranspiler(config=config)

    t.transpile(
        trainer_id=int(os.getenv("PADDLE_TRAINER_ID", 0)),
        trainers=os.getenv("PADDLE_TRAINER_ENDPOINTS", "127.0.0.1:6170"),
        current_endpoint=os.getenv("PADDLE_CURRENT_ENDPOINT", "127.0.0.1:6170"),
        startup_program=startup_prog,
        program=train_prog)
        

def optimizer_setting(params, args):
    ls = params["learning_strategy"]
    step = 1
    bd = [step * e for e in ls["epochs"]]
    base_lr = params["lr"]
    lr = []
    lr = [base_lr * (0.1 ** i) for i in range(len(bd) + 1)]
    print "bd: ", bd
    print "lr_step: ", lr
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(5e-4))

    if args.loss_type in ["dist_softmax", "dist_arcface"]:
        wrapper = DistributedClassificationOptimizer(
            optimizer, params["learning_strategy"]["batch_size"])
    elif args.loss_type in ["softmax", "arcface"]:
        wrapper = optimizer
    return wrapper

def train(args):
    # parameters from arguments
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir

    image_shape = [int(m) for m in args.image_shape.split(",")]

    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
    worker_num = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
    assert model_name in model_list, "{} is not in lists: {}".format(args.model, model_list)

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
        prob_list = fluid.layers.split(prob_all, dim=0, num_or_sections=worker_num)
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

    # initialize optimizer
    optimizer = optimizer_setting(params, args)
    opts = optimizer.minimize(loss)

    if args.loss_type in ["dist_softmax", "dist_arcface"]:
        global_lr = optimizer._optimizer._global_learning_rate()
    elif args.loss_type in ["softmax", "arcface"]:
        global_lr = optimizer._global_learning_rate()

    origin_prog = train_prog.clone()
    transpile(startup_prog, train_prog)
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

    if checkpoint is not None:
        fluid.io.load_persistables(exe, checkpoint)

    if pretrained_model:
        pretrained_model = os.path.join(pretrained_model, str(trainer_id))
        def if_exist(var):
            has_var = os.path.exists(os.path.join(pretrained_model, var.name))
            if has_var:
                print('var: %s found' % (var.name))
            return has_var
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_reader = paddle.batch(reader.arc_train(), batch_size=args.train_batch_size)
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
            label = data[1]
            assert label < args.class_dim, \
                "number of classes of train dataset should be less than the class_dim user specified."
            nsamples += real_batch_size
            t1 = time.time()
            loss, lr, acc1, acc5 = exe.run(train_prog,
                               feed=feeder.feed(data),
                               fetch_list=fetch_list_train,
                               use_program_cache=True)
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
                    embeddings /= np.sqrt((embeddings * embeddings).sum(axis=1))[:, None]
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
    assert args.loss_type in ["softmax", "arcface", "dist_softmax", "dist_arcface"], \
        "Unknown loss type: {}".format(args.loss_type)
    print_arguments(args)
    train(args)

if __name__ == '__main__':
    main()
