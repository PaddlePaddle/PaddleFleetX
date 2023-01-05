model_item=imagen_SR512
dp_degree=1
mp_degree=1
pp_degree=1
bs_item=1
fp_item=fp32
run_mode=DP1-MP1-PP1
device_num=N1C1
yaml_path=ppfleetx/configs/multimodal/imagen/imagen_super_resolution_512.yaml

model=imagen
micro_bs=1

cd ./benchmarks
bash ./test_tipc/imagen/dygraph/benchmark_common/prepare.sh
# run
bash ./test_tipc/imagen/dygraph/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${yaml_path} 2>&1;
