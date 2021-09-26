set -x

export PYTHONPATH=./atarashi/:$PYTHONPATH


export GLOG_v=1
export NCCL_DEBUG=INFO
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit

rm -rf main_*
rm -rf start_*
rm -rf *.prototxt
rm -rf core.*

task_name='gpt3-test'
output_dir=output/${task_name}
#rm -rf ${output_dir}

gpus=${1:-0,1,2,3,4,5,6,7}
global_bsz=${2:-32}
micro_bsz=${3:-8}
num_mp=${4:-2}
num_pp=${5:-4}
num_dp=${6:-1}
debug=${7:-false}

use_sharding=true
num_sharding=1
if [ $((${num_mp} * ${num_pp} * ${num_dp})) -eq 1 ] ; then
    use_sharding=false
fi

python -m paddle.distributed.fleet.launch \
    --gpus=${gpus} \
    --log_dir ${output_dir}/log \
run_pretraining.py \
    --global_bsz ${global_bsz} \
    --micro_bsz ${micro_bsz} \
    --max_seq_len 512 \
    --ernie_config_file ./config/ernie_small_base_config.json \
    --learning_rate 1e-4 \
    --log_steps 1 \
    --num_train_steps 10 \
    --save_steps 10 \
    --output_dir ${output_dir} \
    --use_recompute true \
    --use_sharding ${use_sharding} \
    --num_mp=${num_mp} \
    --num_sharding=1 \
    --num_pp=${num_pp} \
    --num_dp=${num_dp} \
    --debug ${debug} \
    --init_checkpoint ${output_dir}/saved_model_pp${num_pp}mp${num_mp} \
    --init_checkpoint_step -1 \
    --use_quantize false \
    --use_amp true \
    --exported_feeded_var_names src_ids sent_ids pos_ids \
    --exported_fetch_var_names fc_1.tmp_1 \

rm -rf main_*
rm -rf start_*
rm -rf *.prototxt
rm -rf core.*
