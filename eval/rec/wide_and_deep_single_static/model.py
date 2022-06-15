import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


class WideDeepLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(WideDeepLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        self.wide_part = paddle.nn.Linear(
            in_features=self.dense_feature_dim,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0, std=1.0 / math.sqrt(self.dense_feature_dim))))

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        sizes = [sparse_feature_dim * num_field + dense_feature_dim
                 ] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, sparse_inputs, dense_inputs):
        # wide part
        wide_output = self.wide_part(dense_inputs)

        # deep part
        sparse_embs = []
        for s_input in sparse_inputs:
            emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            sparse_embs.append(emb)

        deep_output = paddle.concat(x=sparse_embs + [dense_inputs], axis=1)
        for n_layer in self._mlp_layers:
            deep_output = n_layer(deep_output)

        prediction = paddle.add(x=wide_output, y=deep_output)
        pred = F.sigmoid(prediction)
        return pred


class WideDeepModel:
    def __init__(self, sparse_feature_number=1000001, sparse_inputs_slots=27, sparse_feature_dim=10, dense_input_dim=13, fc_sizes=[400, 400, 400]):
        self.sparse_feature_number = sparse_feature_number
        self.sparse_inputs_slots = sparse_inputs_slots
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_input_dim = dense_input_dim
        self.fc_sizes = fc_sizes

    def net(self, is_train=True):
        dense_input = paddle.static.data(name="dense_input", shape=[
                                         None, self.dense_input_dim], dtype="float32")

        sparse_inputs = [
            paddle.static.data(name="C" + str(i),
                               shape=[None, 1],
                               lod_level=1,
                               dtype="int64") for i in range(1, self.sparse_inputs_slots)
        ]

        label_input = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

        self.inputs = [dense_input] + sparse_inputs + [label_input]

        wide_deep_model = WideDeepLayer(self.sparse_feature_number, self.sparse_feature_dim,
                                        self.dense_input_dim, self.sparse_inputs_slots - 1, self.fc_sizes)

        pred = wide_deep_model.forward(sparse_inputs, dense_input)
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        label_float = paddle.cast(label_input, dtype="float32")

        # loss
        cost = paddle.nn.functional.log_loss(input=pred, label=label_float)
        avg_cost = paddle.mean(x=cost)
        self.loss = avg_cost
