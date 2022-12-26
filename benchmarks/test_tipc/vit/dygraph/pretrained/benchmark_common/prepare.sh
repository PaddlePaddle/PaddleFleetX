python -m pip install -r ../requirements.txt
# get data
cd ../
mkdir dataset && cd dataset
python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
    --remote-path ./plsc_data/ILSVRC2012/ \
    --local-path ./ \
    --mode download
cd -
