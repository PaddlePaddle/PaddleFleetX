model_item=ViT_large_patch16_384_ft_fused_True
fp_item=fp16
bs_item=512
run_mode=DP
device_num=N1C8
use_fused_attn=True
max_iter=1


cd ./benchmarks
bash ./test_tipc/vit/dygraph/finetune/benchmark_common/prepare.sh
# run
bash ./test_tipc/vit/dygraph/finetune/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${bs_item} ${run_mode} ${device_num} \
${use_fused_attn} ${max_iter} 2>&1;
