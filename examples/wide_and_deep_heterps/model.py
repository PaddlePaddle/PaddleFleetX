import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
from paddle.fluid.layers.nn import _pull_box_sparse

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

        #self.embedding = _pull_box_sparse(self.slots, size = self.sparse_feature_dim, is_distributed=True, is_sparse=True)
	"""
	self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))
	"""
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
        #sparse_embs = []
        sparse_embs = _pull_box_sparse(sparse_inputs, size = self.sparse_feature_dim, is_distributed=True, is_sparse=True)
        """
        for s_input in sparse_inputs:
            emb = self.embedding(s_input)
            emb = _pull_box_sparse(, size = self.sparse_feature_dim, is_distributed=True, is_sparse=True)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            sparse_embs.append(emb)
	"""
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

        self._metrics = {}

    def acc_metrics(self, pred, label):
        correct_cnt = paddle.static.create_global_var(
            name="right_cnt", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = paddle.static.create_global_var(
            name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)

        batch_cnt = paddle.sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_accuracy = paddle.static.accuracy(input=pred, label=label)
        batch_correct = batch_cnt * batch_accuracy

        paddle.assign(correct_cnt + batch_correct, correct_cnt)
        paddle.assign(total_cnt + batch_cnt, total_cnt)
        accuracy = correct_cnt / total_cnt

        self._metrics["acc"] = {}
        self._metrics["acc"]["result"] = accuracy
        self._metrics["acc"]["state"] = {
            "total": (total_cnt, "float32"), "correct": (correct_cnt, "float32")}

    def auc_metrics(self, pred, label):
        auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg] = paddle.static.auc(input=pred,
                                                                                                 label=label,
                                                                                                 num_thresholds=2**12,
                                                                                                 slide_steps=20)

        self._metrics["auc"] = {}
        self._metrics["auc"]["result"] = auc
        self._metrics["auc"]["state"] = {"stat_pos": (
            stat_pos, "int64"), "stat_neg": (stat_neg, "int64")}

    def mae_metrics(self, pred, label):
        abserr = paddle.static.create_global_var(
            name="abserr", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = paddle.static.create_global_var(
            name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)

        batch_cnt = paddle.sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_abserr = paddle.nn.functional.l1_loss(
            pred, label, reduction='sum')

        paddle.assign(abserr + batch_abserr, abserr)
        paddle.assign(total_cnt + batch_cnt, total_cnt)
        mae = abserr / total_cnt

        self._metrics["mae"] = {}
        self._metrics["mae"]["result"] = mae
        self._metrics["mae"]["state"] = {
            "total": (total_cnt, "float32"), "abserr": (abserr, "float32")}

    def mse_metrics(self, pred, label):
        sqrerr = paddle.static.create_global_var(
            name="sqrerr", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = paddle.static.create_global_var(
            name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)

        batch_cnt = paddle.sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_sqrerr = paddle.nn.functional.mse_loss(
            pred, label, reduction='sum')

        paddle.assign(sqrerr + batch_sqrerr, sqrerr)
        paddle.assign(total_cnt + batch_cnt, total_cnt)
        mse = sqrerr / total_cnt
        rmse = paddle.sqrt(mse)

        self._metrics["mse"] = {}
        self._metrics["mse"]["result"] = mse
        self._metrics["mse"]["state"] = {
            "total": (total_cnt, "float32"), "sqrerr": (sqrerr, "float32")}

        self._metrics["rmse"] = {}
        self._metrics["rmse"]["result"] = rmse
        self._metrics["rmse"]["state"] = {
            "total": (total_cnt, "float32"), "sqrerr": (sqrerr, "float32")}

    def net(self, is_train=True):
        dense_input = paddle.static.data(name="dense_input", shape=[
                                         None, self.dense_input_dim], dtype="float32")

        sparse_inputs = [
            paddle.static.data(name=str(i),
                               shape=[None, 1],
                               lod_level=1,
                               dtype="int64") for i in range(1, self.sparse_inputs_slots)
        ]

        self.slots_name = [int(i) for i in range(1, self.sparse_inputs_slots)]

        label_input = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

        self.inputs = [dense_input] + sparse_inputs + [label_input]

        wide_deep_model = WideDeepLayer(self.sparse_feature_number, self.sparse_feature_dim,
                                        self.dense_input_dim, self.sparse_inputs_slots - 1, self.fc_sizes)

        pred = wide_deep_model.forward(sparse_inputs, dense_input)
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        label_float = paddle.cast(label_input, dtype="float32")

        with paddle.utils.unique_name.guard():
            self.acc_metrics(pred, label_input)
            self.auc_metrics(predict_2d, label_input)
            self.mae_metrics(pred, label_float)
            self.mse_metrics(pred, label_float)

        # loss
        cost = paddle.nn.functional.log_loss(input=pred, label=label_float)
        avg_cost = paddle.mean(x=cost)
        self.cost = avg_cost
