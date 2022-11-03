import pickle
import numpy as np
import copy
import paddle

def strip(data):
    assert isinstance(data, dict)
    data.pop("StructuredToParameterName@@", None)

def load_params(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding='latin1')
    return data

def parse_rank_auto(dp, mp, pp):
    ranks = list(range(8))
    ranks = ranks[pp*4:pp*4+4]
    ranks = ranks[dp*2:dp*2+2]
    rank = ranks[mp]
    return rank

def test(strategy):
    mp1 = parse_rank_auto(*strategy)
    mp_model = load_params(f"before_train/static_saved_dist{mp1}.pdparams")

    pp0_static = mp_model

    mp1 = parse_rank_dygraph(*strategy)
    mp_model_dy = load_params(f"output_dp2mp2pp2/auto_dist{mp1}.pdparams")

    pp0_dygraph = mp_model_dy

    for k in pp0_static.keys():
        if k in pp0_dygraph:
            d1 = pp0_static[k]
            d2 = pp0_dygraph[k]
            assert np.allclose(d1, d2)
        else:
            print("not matched key:", k)

def parse_rank_dygraph(dp, mp, pp):
    ranks = list(range(4))
    ranks = ranks[pp*2:pp*2+2]
    rank = ranks[mp]
    return rank

if __name__ == "__main__":

   model1 = paddle.load("/tmp/before_dist0.pdparams")

   model2 = paddle.load("output_345/epoch_0_step_0/saved_dist0.pdparams")

   print(model1.keys())
   print(model2.keys())

   for k, v in model1.items():
        print(k)
        # print(v)
        # print(model2[k])
        assert k in model2
        diff = v.numpy() - model2[k].numpy()
        print(k, np.max(diff), np.min(diff))
        # assert np.allclose(v.numpy(), model2[k].numpy()), k
