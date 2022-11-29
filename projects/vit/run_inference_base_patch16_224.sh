echo "step 1: download parameters"
mkdir -p ckpt
wget -O ckpt/model.pdparams https://paddlefleetx.bj.bcebos.com/model/vision/vit/imagenet2012-ViT-B_16-224.pdparams

echo "step 2: export model"
python tools/export.py \
    -c ppfleetx/configs/vis/vit/ViT_base_patch16_224_inference.yaml \
    -o Engine.save_load.ckpt_dir=./ckpt/

echo "step 3: run VIT inference"
python projects/vit/inference.py -c ppfleetx/configs/vis/vit/ViT_base_patch16_224_inference.yaml
