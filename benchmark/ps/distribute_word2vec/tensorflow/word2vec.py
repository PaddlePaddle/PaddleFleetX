# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec mini-batched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does traditional minibatching.

The key ops used are:
* placeholder for feeding in tensors for each example.
* embedding_lookup for fetching rows from the embedding matrix.
* sigmoid_cross_entropy_with_logits to calculate the loss.
* GradientDescentOptimizer for optimizing the loss.
* skipgram custom op that does input processing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
import reader
from net import skip_gram_word2vec


flags = tf.app.flags

flags.DEFINE_string("model_output_dir", 'output/local', "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_data_dir", 'train_data', "Training text data directory.")
flags.DEFINE_integer("embedding_size", 300, "The embedding dimension size.")
flags.DEFINE_integer("epochs_to_train", 5, "Number of epochs to train." 
                     "Each epoch processes the training data once completely.")
flags.DEFINE_float("learning_rate", 1.0, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 5,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 100,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")

flags.DEFINE_string("dict_path", 'thirdparty/test_build_dict', "dict path")
FLAGS = flags.FLAGS

def get_batch(reader, batch_size):
    example_batch = []
    label_batch = []
    for example, label in reader():
        example_batch.append(example)
        label_batch.append(label)
        if len(example_batch) >= batch_size:
            yield example_batch, label_batch
            example_batch = []
            label_batch = []


def print_log(log_str):
    """
    :param log_str: log info
    :return: no return
    """
    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S',
                               time.localtime(time.time()))
    print(str(time_stamp) + " " + log_str)

def GetFileList(data_path):
    return os.listdir(data_path)

def main(_):
    if not os.path.isdir(FLAGS.model_output_dir):
        os.mkdir(FLAGS.model_output_dir)

    with tf.Graph().as_default(), tf.Session() as session:
        global_step = tf.Variable(0, name="global_step")
    
        filelist = GetFileList(FLAGS.train_data_dir)
        word2vec_reader = reader.Word2VecReader(FLAGS.dict_path, FLAGS.train_data_dir,
                                                filelist, 0, 1)
        print_log("dict_size: {}".format(word2vec_reader.dict_size))

        examples, labels, loss = skip_gram_word2vec(word2vec_reader.dict_size, FLAGS.embedding_size, FLAGS.batch_size, np.array(word2vec_reader.id_frequencys), FLAGS.num_neg_samples)
        lr = tf.train.exponential_decay(
                learning_rate=FLAGS.learning_rate, global_step=global_step, decay_steps=100000, decay_rate=0.999, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss, global_step=global_step)
        saver = tf.train.Saver(max_to_keep=None)
        tf.global_variables_initializer().run()
        log_dir = FLAGS.model_output_dir + '/model.ckpt'
        for epoch in xrange(FLAGS.epochs_to_train):
            start_time = time.time()
            for examples_, labels_, in get_batch(word2vec_reader.train(), FLAGS.batch_size):
                feed_dict = {}
                feed_dict[examples] = examples_
                feed_dict[labels] = labels_
                _, loss_, step_ = session.run([train_op, loss, global_step], feed_dict=feed_dict)
                if step_ % 1000 == 0:
                    print("Epoch %4d Step %8d loss = %6.2f" % (epoch, step_, loss_))
            now = time.time()
            print("epoch: %4d total time: %8d" % (epoch, now - start_time))
            saver.save(session, log_dir, global_step=step)
if __name__ == "__main__":
  tf.app.run()
