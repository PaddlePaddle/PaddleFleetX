sed -i "s/paddleslim/#paddleslim/g" ../requirements.txt
rm -rf /usr/local/lib/python3.7/dist-packages/paddleslim*
python -m pip uninstall paddleslim -y
hadoop fs -D fs.default.name=afs://cygnus.afs.baidu.com:9902 -D hadoop.job.ugi=userdata,userdata -get /user/userdata/PaddleSlim.tar.gz ./
tar zxvf PaddleSlim.tar.gz
cd PaddleSlim
python -m pip install -r requirements.txt
python setup.py install
python -m pip list | grep paddleslim
cd -

python -m pip install -r ../requirements.txt
# get data
cd ../
rm -rf data
mkdir data
wget -O data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget -O data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz