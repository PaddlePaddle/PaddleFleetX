rm -rf ./log_auto/
python -m paddle.distributed.launch --log_dir ./log_auto --devices "0,1" \
    ./tools/auto.py \
    -c ./ppfleetx/configs/vis/vit/auto/ViT_base_patch16_384_ft_cifar10_dp2.yaml