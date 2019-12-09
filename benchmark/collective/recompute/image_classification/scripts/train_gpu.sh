
export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_CARDS=4
distributed_args=""
if [[ ${NUM_CARDS} == "1" ]]; then
    distributed_args="--selected_gpus 0"
fi

set -x

python -m paddle.distributed.launch ${distributed_args} --log_dir log \
    train_with_fleet.py \
       --model=ResNet50 \
       --batch_size=780 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --lr_strategy=cosine_decay \
       --lr=0.1 \
       --num_epochs=200 \
       --l2_decay=1.2e-4 \
       --data_dir=/ssd1/liuyi05/imagenet/imagenet_resized \
       --use_recompute=True
