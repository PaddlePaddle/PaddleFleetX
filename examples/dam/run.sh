export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_eager_delete_tensor_gb=0.0

# train on ubuntu
python -u train.py
  --do_train True \
  --filelist train.ubuntu.files \
  --save_path ./model_files/ubuntu \
  --vocab_size 434512 \
  --batch_size 256 \
  --num_scan_data 1 \
  --data_source ubuntu \
  --vocab_path data/ubuntu/word2id

#test on ubuntu
python -u train.py \
  --do_test True \
  --filelist test.ubuntu.files \
  --model_path model_files/ubuntu/model.0 \
  --vocab_path data/ubuntu/word2id \
  --vocab_size 434512 \
  --save_path saved_models/ubuntu \
  --data_source ubuntu \
  --batch_size 32

# train on douban
python -u train.py
  --do_train True \
  --filelist train.douban.files \
  --save_path ./model_files/douban \
  --vocab_size 172130 \
  --vocab_path data/douban/word2id \
  --data_source douban \
  --channel1_num 16 \
  --num_scan_data 1 \
  --batch_size 256

#test on douban
python -u train.py \
  --do_test True \
  --ext_eval \
  --filelist test.douban.files \
  --model_path ./model_files/douban/model.0 \
  --vocab_path data/douban/word2id \
  --save_path saved_models/douban \
  --vocab_size 172130 \
  --channel1_num 16 \
  --data_source douban \
  --batch_size 32
