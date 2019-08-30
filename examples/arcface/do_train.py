import os
import sys
import math
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
from paddle.fluid.transpiler.details.program_utils import program_to_code
from paddle.fluid.transpiler.collective import DistributedClassificationOptimizer

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('train_batch_size', int, 512, "Minibatch size.")
add_arg('test_batch_size', int, 120, "Minibatch size.")
add_arg('num_epochs', int, 120, "number of epochs.")
add_arg('image_shape', str, "3,112,112", "input image size")
add_arg('model_save_dir', str, "output", "model save directory")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('checkpoint', str, None, "Whether to resume checkpoint.")
add_arg('lr', float, 0.1, "set learning rate.")
add_arg('model', str, "ResNet_ARCFACE50", "Set the network to use.")
add_arg('loss', str, "distfc", "Set loss to use.")
add_arg('margin', float, 0.5, "margin.")
add_arg('scale', float, 64.0, "scale.")
add_arg('mode', str, '2', "margin mode.")
add_arg('with_test', bool, False, "open test mode.")
# yapf: enable

model_list = [m for m in dir(resnet) if "__" not in m]

def transpile(startup_prog, train_prog):
    config = fluid.DistributeTranspilerConfig()
    config.mode = 'collective'
    config.collective_mode = 'grad_allreduce'
    t = fluid.DistributeTranspiler(config=config)

    t.transpile(
        trainer_id=int(os.getenv("PADDLE_TRAINER_ID")),
        trainers=os.getenv("PADDLE_TRAINER_ENDPOINTS"),
        current_endpoint=os.getenv("PADDLE_CURRENT_ENDPOINT"),
        startup_program=startup_prog,
        program=train_prog)

def optimizer_setting(params):
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

    wrapper = DistributedClassificationOptimizer(optimizer, params["learning_strategy"]["batch_size"])
    return wrapper

def train(args):
    # parameters from arguments
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir

    image_shape = [int(m) for m in args.image_shape.split(",")]

    trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))

    assert model_name in model_list, "{} is not in lists: {}".format(args.model, model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = resnet.__dict__[model_name]()
    emb, loss = model.net(input=image,
                    label=label,
                    emb_dim=512,
                    class_dim=85184,
                    loss_type=args.loss,
                    margin=args.margin,
                    scale=args.scale)

    startup_prog = fluid.default_startup_program()
    train_prog = fluid.default_main_program()
    test_program = train_prog.clone(for_test=True)

    # parameters from model and arguments
    params = model.params
    params["lr"] = args.lr
    params["num_epochs"] = args.num_epochs
    params["learning_strategy"]["batch_size"] = args.train_batch_size

    # initialize optimizer
    optimizer = optimizer_setting(params)
    opts = optimizer.minimize(loss)

    global_lr = optimizer._optimizer._global_learning_rate()

    origin_prog = train_prog.clone()
    transpile(startup_prog, train_prog)
    if trainer_id == 0:
        with open('start.program', 'w') as fout:
            program_to_code(startup_prog, fout, True)
        with open('main.program', 'w') as fout:
            program_to_code(train_prog, fout, True)
        with open('origin.program', 'w') as fout:
            program_to_code(origin_prog, fout, True)

    gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    place = fluid.CUDAPlace(gpu_id)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if checkpoint is not None:
        fluid.io.load_persistables(exe, checkpoint)

    if pretrained_model:
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

    fetch_list_train = [loss.name, global_lr.name]
    fetch_list_test = [emb.name]

    num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM"))
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
            loss, lr = exe.run(train_prog,
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
                print("Pass:%d batch:%d lr:%f loss:%f qps:%.2f" % (
                    pass_id, batch_id, avg_lr, avg_loss, nsamples / local_time))
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
                            [_embeddings] = exe.run(test_program, fetch_list = fetch_list_test, feed=feeder.feed(_data))
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
        if trainer_id == 0:
            model_path = os.path.join(model_save_dir + '/' + model_name,
                                  str(pass_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            fluid.io.save_persistables(exe, model_path)

def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)

if __name__ == '__main__':
    main()
