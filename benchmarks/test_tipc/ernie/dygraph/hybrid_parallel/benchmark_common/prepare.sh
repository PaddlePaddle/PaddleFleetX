python -m pip install -r ../requirements.txt
# get data
cd ../
rm -rf dataset/ernie
mkdir -p dataset/ernie
unset http_proxy && unset https_proxy
wget -O dataset/ernie/cluecorpussmall_14g_1207_ids.npy \
    http://10.255.129.12:8811/cluecorpussmall_14g_1207_ids.npy
wget -O dataset/ernie/cluecorpussmall_14g_1207_idx.npz \
    http://10.255.129.12:8811/cluecorpussmall_14g_1207_idx.npz