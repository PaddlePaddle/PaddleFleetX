python -m pip install -r ../requirements.txt
# get data
cd ../
mkdir dataset && cd dataset
cp -r ${BENCHMARK_ROOT}/models_data_cfs/Paddle_distributed/ILSVRC2012.tgz ./
tar -zxf ILSVRC2012.tgz
cd -
