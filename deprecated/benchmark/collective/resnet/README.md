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

## How to use DGC
Set USE_DGC=True. Note, use DGC must close fuse&fp16 for now, but we've added this logic
to the code, so you don't need to change the value of FUSE in scripts/train_gpus.sh.
``` bash
USE_DGC=True
DGC_RAMPUP_BEGIN_STEP=5008 # for 32 cards, DGC start from 4 epochs
```

