# ResNet50 and VGG16 on ImageNet with Fleet APIs
## How to change different model
you can change Resnet50 or VGG16 model easily by editing scripts/train_gpu.sh:
```
MODEL=ResNet50  
# MODEL=VGG16
```

## How to run

### with single gpu multiple cards
```
sh run.sh
```


## How to use adaptive local step for local sgd
```
sh run_ada.sh
```

