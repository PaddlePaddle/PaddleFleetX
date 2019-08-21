#!/bin/bash
if [ ! -d "./data" ]; then
  mkdir data
fi
wget https://paddlerec.bj.bcebos.com/word2vec/text.tar --no-check-certificate
tar xvf text.tar
rm text.tar
mv text data/

python preprocess.py --build_dict --build_dict_corpus_dir data/text/ --dict_path data/test_build_dict

python preprocess.py --filter_corpus --dict_path data/test_build_dict --input_corpus_dir data/text --output_corpus_dir data/convert_text8 --min_count 5 --downsample 0.001

wget https://paddlerec.bj.bcebos.com/word2vec/test_mid_dir.tar --no-check-certificate
tar xvf test_mid_dir.tar
rm test_mid_dir.tar

if [ ! -d "./train_data" ]; then
  mkdir train_data
fi
if [ ! -d "./test_data" ]; then
  mkdir test_data
fi

cp ./data/convert_text8/* ./train_data/
cp ./data/test_build_dict ./
cp ./data/test_build_dict_word_to_id_ ./
cp ./data/test_mid_dir/* ./test_data/

