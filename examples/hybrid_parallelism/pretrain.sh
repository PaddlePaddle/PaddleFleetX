set -x

export PYTHONPATH=./atarashi/:$PYTHONPATH


export GLOG_v=1
export NCCL_DEBUG=INFO
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit

rm -rf *.prototxt
rm -rf core.*

task_name='gpt3-test'
output_dir=./output/${task_name}
rm -rf ${output_dir}

python -m paddle.distributed.fleet.launch \
	--gpus="0,1,2,3" \
	--log_dir ${output_dir}/log \
	--run_mode=collective \
run_pretraining.py \
	--global_bsz 8 \
	--micro_bsz 8 \
	--max_seq_len 512 \
	--ernie_config_file config/ernie_base_config.json \
	--learning_rate 1e-4 \
	--log_steps 1 \
	--num_train_steps 11 \
	--save_steps 10 \
	--output_dir ${output_dir} \
	--use_recompute true \
	--use_sharding true \
	--num_mp=2 \
	--num_sharding=1 \
	--num_pp=2 \
	--num_dp=1 \
    --debug false \

