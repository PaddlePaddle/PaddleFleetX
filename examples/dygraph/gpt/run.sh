#wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/train.data.json_ids.npz
#mkdir data
#mv train.data.json_ids.npz data
#
export DATA_DIR=./data
export PYTHONPATH=$PYTHONPATH:../

rm -rf dp2_pp2_mp4
python -m paddle.distributed.launch --log_dir dp2_pp1_mp4 --gpus "0,1,2,3,4,5,6,7" run_pretrain.py \
   --model_type gpt2 \
   --model_name_or_path gpt2-small-en\
   --input_dir "./data"\
   --output_dir "output"\
   --weight_decay 0.01\
   --grad_clip 1.0\
   --max_steps 80000\
   --save_steps 100000\
   --decay_steps 320000\
   --warmup_rate 0.01\
   --batch_size 16\
   --device gpu \
   --use_amp False \
   --max_seq_len 1024 \
   --mp_degree 2 \
   --dp_degree 2 \
   --pp_degree 2 \
   --micro_batch_size 2

