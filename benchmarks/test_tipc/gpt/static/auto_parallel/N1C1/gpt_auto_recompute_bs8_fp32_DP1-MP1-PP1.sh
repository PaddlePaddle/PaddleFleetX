model_item=gpt_auto_recompute
dp_degree=1
mp_degree=1
pp_degree=1
bs_item=8
fp_item=fp32
run_mode=DP1-MP1-PP1
device_num=N1C1
max_iter=500
use_recompute=True

model=gpt
micro_bs=8

cd ./benchmarks
bash ./test_tipc/gpt/static/auto_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/static/auto_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${max_iter} ${use_recompute} 2>&1;
