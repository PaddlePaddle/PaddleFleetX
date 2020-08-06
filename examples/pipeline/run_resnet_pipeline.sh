export PATH=/workspace/software/python/bin:$PATH
export GLOG_v=0
which python

microbatch_size=32
microbatch_num=1
#data_dir='/ssd3/lilong/cifar_10/'
#data_file='cifar-10-python.tar.gz'
#dataset='cifar10'
data_dir='/workspace/train_data'
data_file='train.txt'
dataset='ImageNet'
pass_id=0

rm result_${microbatch_num}_${microbatch_size}.txt
rm *.prototxt
python train_resnet_pipeline.py \
    --model_dir=./saved_model \
    --dataset=${dataset} \
    --passid=${pass_id} \
    --data_file=${data_file} \
    --data_dir=${data_dir} \
    --microbatch_size=${microbatch_size} \
    --microbatch_num=${microbatch_num} > result_${microbatch_num}_${microbatch_size}.txt 2>&1
    
