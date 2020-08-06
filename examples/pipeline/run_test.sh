export PATH=/home/lilong/workspace/model_parallel/python/bin:$PATH
export GLOG_v=0
which python

data_dir='/ssd3/lilong/cifar_10/'
data_file='cifar-10-python.tar.gz'
dataset='cifar10'


rm result_test.txt
for pass_id in {0}; do
    python test.py \
        --data_dir=${data_dir} \
        --passid=${pass_id} \
        --model_dir=./saved_model \
        --dataset=$dataset \
        --data_file=${data_file} >> result_test.txt 2>&1
done
