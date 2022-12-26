model_item=imagen_397M_text2im_64
dp_degree=8
mp_degree=1
pp_degree=1
bs_item=8
fp_item=fp32
run_mode=DP8-MP1-PP1
device_num=N1C8
yaml_path=ppfleetx/configs/multimodal/imagen/imagen_397M_text2im_64x64.yaml
max_iter=1000
num_train_epochs=5

model=imagen
micro_bs=1

cd ./benchmarks
bash ./test_tipc/imagen/dygraph/benchmark_common/prepare.sh
# run
bash ./test_tipc/imagen/dygraph/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${yaml_path} ${max_iter} ${num_train_epochs} 2>&1;
