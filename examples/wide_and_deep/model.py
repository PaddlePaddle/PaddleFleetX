import paddle


def net(self, args, batch_size=4, lr=0.01):
    """
    network definition

    Args:
        batch_size(int): the size of mini-batch for training
        lr(float): learning rate of training
    Returns:
        avg_cost: LoDTensor of cost.
    """

    dnn_input_dim, lr_input_dim = 10, 10
    dnn_layer_dims = [128, 64, 32]

    dnn_data = fluid.layers.data(
        name="dnn_data",
        shape=[-1, 1],
        dtype="int64",
        lod_level=1,
        append_batch_size=False)
    lr_data = fluid.layers.data(
        name="lr_data",
        shape=[-1, 1],
        dtype="int64",
        lod_level=1,
        append_batch_size=False)
    label = fluid.layers.data(
        name="click",
        shape=[-1, 1],
        dtype="int64",
        lod_level=0,
        append_batch_size=False)

    datas = [dnn_data, lr_data, label]

    if args.reader == "pyreader":
        self.reader = fluid.io.PyReader(
            feed_list=datas,
            capacity=64,
            iterable=False,
            use_double_buffer=False)

    init = fluid.initializer.Normal()

    dnn_embedding = fluid.contrib.layers.sparse_embedding(
        input=dnn_data,
        size=[dnn_input_dim, dnn_layer_dims[0]],
        is_test=inference,
        param_attr=fluid.ParamAttr(
            name="deep_embedding", initializer=init))

    dnn_pool = fluid.layers.sequence_pool(
        input=dnn_embedding, pool_type="sum")

    dnn_out = dnn_pool

    for i, dim in enumerate(dnn_layer_dims[1:]):
        fc = fluid.layers.fc(
            input=dnn_out,
            size=dim,
            act="relu",
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01)),
            name='dnn-fc-%d' % i)
        dnn_out = fc

    # build lr model
    lr_embbding = fluid.contrib.layers.sparse_embedding(
        input=lr_data,
        size=[lr_input_dim, 1],
        is_test=inference,
        param_attr=fluid.ParamAttr(
            name="wide_embedding",
            initializer=fluid.initializer.Constant(value=0.01)))

    lr_pool = fluid.layers.sequence_pool(input=lr_embbding, pool_type="sum")
    merge_layer = fluid.layers.concat(input=[dnn_out, lr_pool], axis=1)
    predict = fluid.layers.fc(input=merge_layer, size=2, act='softmax')

    acc = fluid.layers.accuracy(input=predict, label=label)
    auc_var, _, _ = fluid.layers.auc(input=predict, label=label)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    return feeds, predict, avg_cost

