export CUDA_VISIBLE_DEVICES=0,1

# alexnet_dygraph_pipeline.py是用户运行动态图流水线的python文件
python -m paddle.distributed.launch alexnet_dygraph_pipeline.py 
