echo "start..."
# User configurations
HADOOP_FS_NAME=afs://xingtian.afs.baidu.com:9902
HADOOP_UGI=Paddle_Data,Paddle_gpu@2017

echo "fetch gcc..."
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/gcc-4.8.2.tar.gz ./
mkdir /opt/compiler
tar zxvf gcc-4.8.2.tar.gz > /dev/null
mv gcc-4.8.2 /opt/compiler
rm gcc-4.8.2.tar.gz

echo "fetch python..."
hadoop fs -D fs.default.name=${HADOOP_FS_NAME} -D hadoop.job.ugi=${HADOOP_UGI} -get ./lilong/python_distclassify.tar.gz ./
echo "untar..."
tar zxvf python_distclassify.tar.gz > /dev/null
rm python_distclassify.tar.gz

echo "fetch CUDA environment"
hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/cuda-9.2.tar.gz ./
echo "untar..."
tar zxvf cuda-9.2.tar.gz > /dev/null
rm cuda-9.2.tar.gz

hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/cudnn742c92.tgz ./
echo "untar..."
tar zxvf cudnn742c92.tgz > /dev/null
rm cudnn742c92.tgz

hadoop fs -D fs.default.name=afs://xingtian.afs.baidu.com:9902 -D hadoop.job.ugi=Paddle_Data,Paddle_gpu@2017 -get ./lilong/nccl2.3.7_cuda9.2.tar.gz ./
echo "untar..."
tar zxvf nccl2.3.7_cuda9.2.tar.gz > /dev/null
rm nccl2.3.7_cuda9.2.tar.gz

pushd thirdparty

echo "list folder $PWD"
ls

filelist=(MS1M.tar)
for file in ${filelist[@]}; do
  tar xf $file > /dev/null
  rm $file
done

popd


export LD_LIBRARY_PATH=`pwd`/cuda-9.2/lib64:`pwd`/cudnn742c92/lib64:`pwd`/nccl2.3.7_cuda9.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/cuda-9.2/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO

export PATH=`pwd`/python/bin:$PATH
export PYTHONPATH=`pwd`/python/lib/python2.7/site-packages:$PYTHONPATH

if [ $PADDLE_TRAINERS ];then
   config="--cluster_node_ips=${PADDLE_TRAINERS} --node_ip=${POD_IP} "
else
    config=" "
fi

selected_gpus="0,1,2,3,4,5,6,7"

export FLAGS_cudnn_exhaustive_search=true 
export FLAGS_fraction_of_gpu_memory_to_use=0.86 
export FLAGS_eager_delete_tensor_gb=0.0 

mv thirdparty/MS1M ./train_data

python -m paddle.distributed.launch $config \
  --selected_gpus $selected_gpus \
  --log_dir mylog_distfc \
  do_train.py \
  --model=ResNet_ARCFACE50 \
  --train_batch_size=128 \
  --loss=dist_softmax \
  --margin=0.5 \
  --with_test=True

python -m paddle.distributed.launch $config \
  --selected_gpus $selected_gpus \
  --log_dir mylog_distfc \
  do_train.py \
  --model=ResNet_ARCFACE50 \
  --train_batch_size=128 \
  --loss=dist_arcface \
  --margin=0.5 \
  --with_test=True

