import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet


def build_train_net(is_distributed):
    return _build_net(False, is_distributed)


def build_test_net():
    return _build_net(True)


def _build_net(is_test, is_distributed=False):
    with fluid.unique_name.guard():
        image = fluid.layers.data(name='image', shape=[784], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        fc0 = fluid.layers.fc(image, size=128, act='relu')
        CLASS_NUM = 10
        fc1 = fluid.layers.fc(fc0, size=CLASS_NUM)
        

        if is_test:
            softmax = fluid.layers.softmax(fc1)
            acc = fluid.layers.accuracy(softmax, label=label, k=1)
            return [image, label], [acc]
        else:
            if is_distributed:
                all_fc0 = fluid.layers.collective._c_allgather(fc0, fleet.worker_num(), use_calc_stream=True)
            else:
                all_fc0 = fc0
            diff = fc0 - fluid.layers.reduce_mean(all_fc0, dim=0)
            mse = fluid.layers.reduce_mean(diff * diff, dim=1)

            cross_entropy = fluid.layers.softmax_with_cross_entropy(fc1, label)
            cost = 0.5 * (0 - mse) + 0.5 * cross_entropy

            loss = fluid.layers.reduce_mean(cost)
            return [image, label], [loss]
