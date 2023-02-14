python -m pip install -r ../requirements.txt
# get data
cd ../
mkdir dataset && cd dataset
cp -r ${BENCHMARK_ROOT}/models_data_cfs/Paddle_distributed/ILSVRC2012.tgz ./
tar -zxf ILSVRC2012.tgz
cd -

# pretrained
mkdir -p pretrained/vit/
wget -O ./pretrained/vit/imagenet21k-ViT-L_16.pdparams \
https://paddle-wheel.bj.bcebos.com/benchmark/imagenet21k-ViT-L_16.pdparams
