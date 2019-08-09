# ResNet50 on ImageNet with Fleet APIs
## How to Run
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
