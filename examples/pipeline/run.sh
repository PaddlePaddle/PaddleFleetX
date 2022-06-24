export CUDA_VISIBLE_DEVICES=0,1

# alexnet_dygraph_pipeline.py is the python file where the user runs the dynamic graph pipeline
python -m paddle.distributed.launch alexnet_dygraph_pipeline.py 
