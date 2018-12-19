import paddle.fluid as fluid

def mlp(concat, lr_x=1.0):
    fc_layers_input = [concat]
    fc_layers_size = [128, 64, 32, 16, 8]
    fc_layers_act = ["relu"] * (len(fc_layers_size) - 1) + [None]

    for i in range(len(fc_layers_size)):
        fc = fluid.layers.fc(
                input = fc_layers_input[-1],
                size = fc_layers_size[i],
                act = fc_layers_act[i],
                param_attr = fluid.ParamAttr(learning_rate=lr_x))
        fc_layers_input.append(fc)
    return fc_layers_input[-1]

def example_net(user,
                item,
                label,
                dict_dim = 10, #Don't worry, it's fake, doesn't matter
                emb_dim=9):
    """
    example net
    """
    # embedding
    item_emb = []
    for data in item:
        emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim], is_sparse = True, is_distributed=True, param_attr=fluid.ParamAttr(name="embedding"))
        bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
        item_emb.append(bow)
    user_emb = []
    for data in user:
        emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim], is_sparse = True, is_distributed=True, param_attr=fluid.ParamAttr(name="embedding"))
        bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
        user_emb.append(bow)

    concat_user = fluid.layers.concat(user_emb, axis=1)
    concat_item = fluid.layers.concat(item_emb, axis=1)
    fc_user = mlp(concat_user)
    fc_item = mlp(concat_item)

    similarity = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(fc_user, fc_item), dim=1, keep_dim=True)
    prob = fluid.layers.sigmoid(similarity)

    cost = fluid.layers.log_loss(input=prob, label=fluid.layers.cast(x=label, dtype='float32'))

    avg_cost = fluid.layers.mean(x=cost)

    return avg_cost, similarity 

