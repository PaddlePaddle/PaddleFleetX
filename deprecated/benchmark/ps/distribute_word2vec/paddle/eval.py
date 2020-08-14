from model import word2vec
import paddle.fluid as fluid
from argument import params_args
import os

params = params_args()
model = word2vec()
model_path = params.test_model_dir
result = {}

files = os.listdir(model_path)
for model_dir in files:
    epoch = model_dir.split('_')[-1]
    print "process %s" % model_dir
    model.run_infer(params, os.path.join(model_path, model_dir))
