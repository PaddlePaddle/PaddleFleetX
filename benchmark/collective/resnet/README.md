# ResNet50 and VGG16 on ImageNet with Fleet APIs
## How to change different model
you can change Resnet50 or VGG16 model easily by editing scripts/train_gpu.sh:
```
MODEL=ResNet50  
# MODEL=VGG16
```

## How to run
### with single gpu single card
```
scripts/train_gpu.sh:
set NUM_CARDS=1

sh scripts/train_gpu.sh
```

### with single gpu multiple cards
```
scripts/train_gpu.sh:
set NUM_CARDS=8 (your gpu cards number)

sh scripts/train_gpu.sh
```
