model_item=ernie
dp_degree=4
mp_degree=8
pp_degree=1
bs_item=16
fp_item=fp16
run_mode=DP4-MP8-PP1
device_num=N4C32

model=ernie
micro_bs=4

cd ./benchmarks
bash ./test_tipc/ernie/dygraph/hybrid_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/ernie/dygraph/hybrid_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} 2>&1;
