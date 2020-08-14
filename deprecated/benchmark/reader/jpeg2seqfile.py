"""
Convert jpeg images into sequence file.
Organize your directory in the following format:

    ImageNet
           | -- train
                    | -- dir1
                           | -- 1.jpeg
                           | -- ...
                    | -- dir2
                           | -- 10.jpeg
                           | -- ...
           | -- train.txt

    Every line in train.txt represents a space-separated image path and its corresponding label.
    For example:  dir1/1.jpeg 0
"""
from __future__ import print_function
import os
import sys
import cPickle
import visreader.misc.kvtool as kvtool


def jpeg2seqfile(file_list, output):
    """
    Args:
        file_list: Each line of the file represents a space-separated image path and its corresponding label.
        output: The output file.
    """
    with open(output, 'wb') as f_out:
        f_w = kvtool.SequenceFileWriter(f_out)
        count = 0
        with open(file_list, "r") as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                try:
                    img_file, label = line.split(" ")
                    k = os.path.basename(img_file)
                    img_file = os.path.join("train", img_file).replace('JPEG', 'jpeg')
                    with open(img_file, 'r') as img_f:
                        o = {'image': img_f.read(), 'label': int(label)}

                except Exception as e:
                    print('Invalid input line (%s)' % line)
                    print('Expected "[image_path] [label]"')

                f_w.write(k, cPickle.dumps(o, -1))
                count += 1

        print('write %d records to %s' % (count, output))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python %s [file_list] [output]' % (sys.argv[0]))
        exit(1)

    file_list = sys.argv[1]
    output = sys.argv[2]
    assert os.path.isfile(file_list), 'Invalid input file (%s)' % file_list
    assert not os.path.exists(output), 'Output file (%s) already exist.' % output

    jpeg2seqfile(file_list, output)
