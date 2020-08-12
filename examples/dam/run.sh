export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_eager_delete_tensor_gb=0.0

# train on ubuntu
python -u train.py \
  --do_train True \
  --train_data_path data/ubuntu/train.txt \
  --valid_data_path data/ubuntu/valid.txt \
  --word_emb_init data/ubuntu/word_embedding.pkl \
  --vocab_path data/ubuntu/word2id \
  --data_source ubuntu \
  --save_path ./model_files/ubuntu \
  --vocab_size 434512 \
  --batch_size 1 \
  --num_scan_data 1

#test on ubuntu
python -u train.py \
  --do_test True \
  --test_data_path data/ubuntu/test.txt \
  --vocab_path data/ubuntu/word2id \
  --data_source ubuntu \
  --model_path ./model_files/ubuntu/model.epoch_0.step_6000 \
  --save_path model_files/ubuntu/result \
  --vocab_size 434512 \
  --batch_size 512

# train on douban
python -u train.py
  --do_train True \
  --train_data_path data/douban/train.txt \
  --valid_data_path data/douban/dev.txt \
  --word_emb_init data/douban/word_embedding.pkl \
  --vocab_path data/douban/word2id \
  --data_source douban \
  --save_path ./model_files/douban \
  --vocab_size 172130 \
  --channel1_num 16 \
  --batch_size 16 \
  --num_scan_data 1

#test on douban
python -u train.py \
  --do_test True \
  --ext_eval \
  --test_data_path data/douban/test.txt \
  --vocab_path data/douban/word2id \
  --data_source douban \
  --model_path ./model_files/douban/model.epoch_0.step_6000 \
  --save_path model_files/douban/result \
  --vocab_size 172130 \
  --channel1_num 16 \
  --batch_size 512
