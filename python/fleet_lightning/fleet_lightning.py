import argparse


def parse_train_configs():
    parser = argparse.ArgumentParser("fleet-lightning")
    parser.add_argument(
        "--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
        help="training gpu")
    parser.add_argument(
        "--lr", type=float, default=0.00001,
        help="base learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.99,
        help="momentum value")
    args = parser.parse_args()
    return args



    
