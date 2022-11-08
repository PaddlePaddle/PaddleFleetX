log_dir="imagen_log/auto_save"
mkdir -p $log_dir
rm -rf $log_dir/*
python -m paddle.distributed.launch --log_dir $log_dir --devices "0" \
    ./tools/auto.py \
    -c ppfleetx/configs/multimodal/imagen/auto/imagen_super_resolusion_512.yaml \
    -o Model.cond_drop_prob=0. \
    -o Engine.save_load.save_model=False \
    -o Engine.mix_precision.level="" \
    -o Engine.max_steps=51 \
    -o Global.local_batch_size=2 \
    -o Global.micro_batch_size=2
