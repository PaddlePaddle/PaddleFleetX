export FLAGS_USE_STANDALONE_EXECUTOR=False
export CUDA_VISIBLE_DEVICES=1
python ./tools/auto.py -c ./ppfleetx/configs/vis/vit/auto/ViT_base_patch16_384_ft_cifar10_1n8c_dp_fp16o2.yaml