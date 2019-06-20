"""
Encoding jpeg images using base64.
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

The script reads the file 'train.txt'. For each line, the script reads the content of the image and encodes it using
base64. Then, the script combines the base64-encoded image and its label into one line separated by the
tab character (/t), which is finally written to the output file.
"""
from __future__ import print_function
import os
import sys
import pybase64 as base64


def jpeg2base64(file_list, output):
    """
    Args:
        file_list: Each line of the file represents a space-separated image path and its corresponding label.
        output: The output file.
    """
    with open(output, 'wb') as f_out:
        with open(file_list, "r") as f_in:
            count = 0
            for line in f_in.readlines():
                line = line.rstrip('\n')
                try:
                    image_path, label = line.split(" ")
                    image_path = os.path.join("train", image_path).replace('JPEG', 'jpeg')
                    with open(image_path, 'rb') as img_f:
                        o = base64.b64encode(img_f.read(), '-_')

                except Exception as e:
                    print('Invalid input line (%s)' % line)
                    print('Expected "[image_path] [label]"')

                f_out.writelines([o, '\t', label, '\n'])
                count += 1
        print('write %d records to %s' % (count, output))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python %s image_file_list output_file' % (sys.argv[0]))
        exit(1)

    file_list = sys.argv[1]
    output_file = sys.argv[2]
    assert os.path.isfile(file_list), 'Invalid input file (%s).' % file_list
    assert not os.path.exists(output_file), 'Output file (%s) already exist.' % output_file

    jpeg2base64(file_list, output_file)
