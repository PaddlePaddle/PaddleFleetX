export CUDA_VISIBLE_DEVICES=0
rm -rf log
fleetrun --log_dir log \
  train.py \
    --batch_size 128 \
    --use_amp True
