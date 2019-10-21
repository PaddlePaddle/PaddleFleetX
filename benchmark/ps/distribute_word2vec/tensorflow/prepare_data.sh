#! /bin/bash

# download train_data
mkdir data
wget https://paddlerec.bj.bcebos.com/word2vec/1-billion-word-language-modeling-benchmark-r13output.tar
tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar
mv 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ data/

# preprocess data
python preprocess.py --build_dict --build_dict_corpus_dir data/training-monolingual.tokenized.shuffled --dict_path data/test_build_dict
python preprocess.py --filter_corpus --dict_path data/test_build_dict --input_corpus_dir data/training-monolingual.tokenized.shuffled --output_corpus_dir data/convert_text8 --min_count 5 --downsample 0.001
mkdir thirdparty
mv data/test_build_dict thirdparty/
mv data/test_build_dict_word_to_id_ thirdparty/

python preprocess.py --data_resplit --input_corpus_dir=data/convert_text8 --output_corpus_dir=train_data

# download test data
wget https://paddlerec.bj.bcebos.com/word2vec/test_dir.tar
tar xzvf test_dir.tar
mv data/test_dir test_data/
rm -rf data/



