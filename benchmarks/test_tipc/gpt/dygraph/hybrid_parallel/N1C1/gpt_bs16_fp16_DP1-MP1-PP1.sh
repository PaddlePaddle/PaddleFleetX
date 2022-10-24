model_item=gpt
dp_degree=1
mp_degree=1
pp_degree=1
bs_item=16
fp_item=fp16
run_mode=DP1-MP1-PP1
device_num=N1C1

model=gpt
micro_bs=${bs_item}

cd ./benchmarks
bash ./test_tipc/gpt/dygraph/hybrid_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/dygraph/hybrid_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} 2>&1;
