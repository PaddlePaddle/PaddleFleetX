export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_eager_delete_tensor_gb=0.0

# train on ubuntu
python -u train.py \
  --do_train True \
  --train_data_path data_small/ubuntu/train.txt \
  --valid_data_path data_small/ubuntu/valid.txt \
  --vocab_path data/ubuntu/word2id \
  --data_source ubuntu \
  --save_path ./model_files/ubuntu \
  --vocab_size 434512 \
  --batch_size 16 \
  --num_scan_data 1
exit 0

#test on ubuntu
python -u train.py \
  --do_test True \
  --test_data_path data/ubuntu/test.txt \
  --vocab_path data/ubuntu/word2id \
  --data_source ubuntu \
  --model_path model_files/ubuntu/model.4 \
  --save_path model_files/ubuntu/result \
  --vocab_size 434512 \
  --batch_size 512

# train on douban
python -u train.py
  --do_train True \
  --train_data_path data/douban/train.txt \
  --valid_data_path data/douban/dev.txt \
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
  --test_data_path data/douban/dev.txt \
  --vocab_path data/douban/word2id \
  --data_source douban \
  --model_path model_files/douban/model.4 \
  --save_path model_files/douban/result \
  --vocab_size 172130 \
  --channel1_num 16 \
  --batch_size 512
