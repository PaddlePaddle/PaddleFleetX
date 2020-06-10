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
    parser.add_argument(
        "--image_shape", type=str, default="3,224,224",
        help="image shape")
    parser.add_argument(
        "--use_mixup", type=bool, default=False,
        help="Whether to use label_smoothing or not")
    parser.add_argument(
        "--use_dali", type=bool, default=False,
        help="use DALI for preprocess or not.")
    parser.add_argument(
        "--image_mean", type=float, nargs='+',default=[0.485, 0.456, 0.406],
        help="The mean of input image data")
    parser.add_argument(
        "--image_std", type=float, nargs='+',default=[0.229, 0.224, 0.225],
        help="The std of input image data")
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Minibatch size per device.")
    parser.add_argument(
        "--use_gpu", type=bool, default=True,
        help="Whether to use GPU or not.")
    parser.add_argument(
        "--lower_scale", type=float, default=0.08,
        help="Set the lower_scale in ramdom_crop")
    parser.add_argument(
        "--lower_ratio", type=float, default=3./4,
        help="Set the lower_ratio in ramdom_crop")
    parser.add_argument(
        "--upper_ratio", type=float, default=4./3,
        help="Set the upper_ratio in ramdom_crop")
    args = parser.parse_args()
    return args



    
