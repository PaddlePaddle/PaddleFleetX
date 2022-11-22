device=$1
distil_weight=$2
prefix=$3
need_weights=$4
T=$5
kd_rejection=$6


echo $1
echo $2
echo $3
echo $4
echo $5
echo $6


log_dir=1.3B_logs/${prefix}_${distil_weight}
output_dir=1.3B_output/${prefix}_${distil_weight}


mkdir -p $log_dir
mkdir -p $output_dir

export CUDA_VISIBLE_DEVICES=$device

python -m paddle.distributed.launch \
    --log_dir $log_dir \
    ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/distill_1.3B_gpt_345M_single_card.yaml \
    -o Engine.distil_weight=${distil_weight}  -o Engine.save_load.output_dir=$output_dir \
    -o Model.need_weights=${need_weights} -o Engine.T=$T -o Engine.kd_rejection=$kd_rejection
    #-o Optimizer.lr.max_lr=4e-6 -o Optimizer.lr.min_lr=4e-7 -o Engine.max_steps=30000
