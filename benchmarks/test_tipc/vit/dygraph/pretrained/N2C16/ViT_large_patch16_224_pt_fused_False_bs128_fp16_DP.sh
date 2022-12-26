model_item=ViT_large_patch16_224_pt_fused_False
fp_item=fp16
bs_item=128
run_mode=DP
device_num=N2C16
use_fused_attn=False
max_iter=1


cd ./benchmarks
bash ./test_tipc/vit/dygraph/pretrained/benchmark_common/prepare.sh
# run
bash ./test_tipc/vit/dygraph/pretrained/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${bs_item} ${run_mode} ${device_num} \
${use_fused_attn} ${max_iter} 2>&1;
