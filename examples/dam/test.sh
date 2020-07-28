export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_eager_delete_tensor_gb=0.0

#test on ubuntu
python -u train.py \
  --do_test True \
  --filelist test.ubuntu.files \
  --model_path model_files/ubuntu/model.0 \
  --vocab_path data/ubuntu/word2id \
  --vocab_size 434512 \
  --data_source ubuntu \
  --batch_size 256

"""
#test on douban
python -u main.py \
  --do_test True \
  --use_cuda \
  --ext_eval \
  --data_path ./data/douban/data_small.pkl \
  --save_path ./model_files/douban/step_31 \
  --model_path ./model_files/douban/step_31 \
  --vocab_size 172130 \
  --channel1_num 16 \
  --batch_size 32
"""
