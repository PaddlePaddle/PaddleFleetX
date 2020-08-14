from __future__ import print_function
from __future__ import absolute_import
import argparse
import time

parser = argparse.ArgumentParser(description='Benchmarks for various readers.')
parser.add_argument('type',
                    type=str,
                    default='pil',
                    help='The reader type to benchmark.')
parser.add_argument('--data_dir',
                    type=str,
                    default='data/ILSVRC2012',
                    help='Directory for ImageNet dataset.')
parser.add_argument('--num_threads',
                    type=int,
                    default=1,
                    help='Number of thread used to run.')
args = parser.parse_args()


def main(args):
    print("Run benchmark for reader with {}.".format(args.type))
    if args.type.lower() == 'pil':
        from reader_pil import train
    elif args.type.lower() == 'cv2':
        from reader_cv2 import train
    elif args.type.lower() == 'visreader':
        from reader_visreader import train
    elif args.type.lower() == 'libjpeg':
        from reader_libjpeg import train
    elif args.type.lower() == 'base64':
        from reader_libjpeg_turbo_base64 import train
    else:
        print("Unknown reader type: {}.".format(args.type))
        exit()

    begin = time.time()
    reader = train(data_dir=args.data_dir, num_threads=args.num_threads)
    batch_id = 0
    for _ in reader():
        batch_id += 1
        if batch_id % 1000 == 0:
            print("batch_id: {}".format(batch_id))
        if batch_id > 1281167:
            print("why exceed range")
            break
    elapsed = time.time() - begin
    print("Elapsed time: {}, batches: {}, performance: {} batches per second.".format(elapsed, batch_id, batch_id / elapsed))


if __name__ == '__main__':
    main(args)
