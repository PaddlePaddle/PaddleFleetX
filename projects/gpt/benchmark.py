import os
import time
import argparse
import numpy as np

import paddle
import paddle.distributed.fleet as fleet
from ppfleetx.core.engine.inference_engine import InferenceEngine
import custom_setup_ops

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", default=128, type=int, required=False, help="seq length of inputs")
    parser.add_argument("--iter", default=10, type=int,  help="run iterations for timing")
    parser.add_argument("--mp_size", default=1, type=int,  help="")
    parser.add_argument("--model_dir", default="output", type=str,  help="model directory")

    args = parser.parse_args()
    return args

def predict(engine, data, args):

    with engine._static_guard:
        for d, name in zip(data, engine.input_names()):
            handle = engine.predictor.get_input_handle(name)
            handle.copy_from_cpu(d)
        
        for _ in range(1):
            engine.predictor.run()
        engine.predictor.get_output_handle(engine.output_names()[0]).copy_to_cpu()
        
        start = time.perf_counter()
        for _ in range(args.iter):
            engine.predictor.run()
        end = time.perf_counter()
        print(f"batch {args.iter} run time: {1000 * (end - start) / args.iter}ms")
        
        return {name: engine.predictor.get_output_handle(name).copy_to_cpu() \
                for name in engine.output_names()}


def main():
    
    args = parse_args()

    fleet.init(is_collective=True)
    infer_engine = InferenceEngine(args.model_dir, args.mp_size)
    ids = [100] * args.seq_len
    
    # run test
    for batch in [1, 2, 4, 8, 16]:

        whole_data = [ids] * batch
        whole_data = np.array(whole_data, dtype="int64").reshape(1, batch, -1)
        
        _ = predict(infer_engine, whole_data, args)
    

if __name__ == "__main__":
    main()