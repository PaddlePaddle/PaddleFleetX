python -m pip install -r ../requirements.txt
# get data
cd ../
rm -rf data
mkdir data
wget -O data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget -O data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
