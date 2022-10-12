export FLAGS_USE_STANDALONE_EXECUTOR=False
export CUDA_VISIBLE_DEVICES=1

rm -rf ./log_auto/
#   -m paddle.distributed.launch --log_dir=log_auto
python -m paddle.distributed.launch --log_dir=log_auto \
    ./tools/auto.py \
    -c ./ppfleetx/configs/vis/vit/auto/ViT_base_patch16_384_ft_cifar10_single.yaml