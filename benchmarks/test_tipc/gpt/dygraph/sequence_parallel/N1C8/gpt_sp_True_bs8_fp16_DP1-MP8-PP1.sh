model_item=gpt_sp_True
dp_degree=1
mp_degree=8
pp_degree=1
bs_item=8
fp_item=fp16
run_mode=DP1-MP8-PP1
device_num=N1C8
max_iter=1000
sequence_parallel=True

model=gpt
micro_bs=8

cd ./benchmarks
bash ./test_tipc/gpt/dygraph/sequence_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/dygraph/sequence_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${max_iter} ${sequence_parallel} 2>&1;
