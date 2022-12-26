python -m pip install -r ../requirements.txt
# get data
cd ../
mkdir dataset && cd dataset
python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
    --remote-path ./plsc_data/ILSVRC2012/ \
    --local-path ./ \
    --mode download
cd -

# pretrained
mkdir -p pretrained/vit/
wget -O ./pretrained/vit/imagenet21k-ViT-L_16.pdparams \
https://paddle-wheel.bj.bcebos.com/benchmark/imagenet21k-ViT-L_16.pdparams
