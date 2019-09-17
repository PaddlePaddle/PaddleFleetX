#!/usr/bin/env python
# coding=utf-8

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import time
import sys
import os
import glob
import json
from datetime import date, timedelta

import tensorflow as tf

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("num_threads", 11, "thread nums")
tf.app.flags.DEFINE_integer("embedding_size", 10, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 1000, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_integer("dict_size", 1000001, "dict size of sparse feature")
tf.app.flags.DEFINE_integer("dense_nums", 13, "dense feature num")
tf.app.flags.DEFINE_integer("slot_nums", 26, "sparse feature num")

tf.app.flags.DEFINE_string("train_data_dir", 'train_data', "train data dir")
tf.app.flags.DEFINE_string("test_data_dir", 'test_data', "test data dir")
tf.app.flags.DEFINE_string("model_dir", 'output', "model check point dir")


def print_log(log_str):
    """
    :param log_str: log info
    :return: no return
    """
    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S',
                               time.localtime(time.time()))
    print(str(time_stamp) + " " + log_str)

def get_file_list(is_train):
    """
    :param is_train: True for training, and False for testing.
    :return: File list for training or testing.
    """
    if is_train:
        data_dir = FLAGS.train_data_dir
    else:
        data_dir = FLAGS.test_data_dir
    data_files = os.listdir(data_dir)
    file_list = list()
    for data_file in data_files:
        file_list.append(data_dir + '/' + data_file)
    print_log("File list:" + str(file_list))
    print_log("There are a total of %d files" % len(file_list))
    return file_list


C_COLUMNS = ['I' + str(i) for i in range(1, 14)]
D_COLUMNS = ['C' + str(i) for i in range(14, 40)]
LABEL_COLUMN = "label"
CSV_COLUMNS = [LABEL_COLUMN] + C_COLUMNS + D_COLUMNS
# Columns Defaults
CSV_COLUMN_DEFAULTS = [[0]]
C_COLUMN_DEFAULTS = [[0.0] for i in range(FLAGS.dense_nums)]
D_COLUMN_DEFAULTS = [[0] for i in range(FLAGS.slot_nums)]
CSV_COLUMN_DEFAULTS = CSV_COLUMN_DEFAULTS + C_COLUMN_DEFAULTS + D_COLUMN_DEFAULTS

def input_fn(filenames, num_epochs, batch_size=1):
    """
    :param filenames: File list for training or tesing
    :param num_epochs: epoch nums
    :param batch_size: batch size
    :return iterator of dataset
    """
    def parse_csv(line):
        columns = tf.decode_csv(line, record_defaults=CSV_COLUMN_DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        return features

    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(parse_csv, num_parallel_calls=FLAGS.num_threads).prefetch(buffer_size=batch_size * 10)

    dataset = dataset.shuffle(buffer_size=10*batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    return features

def main(_):
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.Variable(0, name='global_step',
                                  trainable=False)
        train_file_list = get_file_list(True)
        print("train_file_list: %s" % str(train_file_list))
        features = input_fn(train_file_list, FLAGS.num_epochs, FLAGS.batch_size)
        embeddings = tf.get_variable("emb", [FLAGS.dict_size, FLAGS.embedding_size], tf.float32,
                initializer=tf.random_uniform_initializer(-1.0,1.0))
        words = []
        for i in range(14, 40):
            key = 'C' + str(i)
            words.append(tf.nn.embedding_lookup(embeddings, features[key]))
        for i in range(1, 14):
            key = 'I' + str(i)
            words.append(tf.reshape(features[key], [-1, 1]))
        concat = tf.concat(words, axis=1)
        label_y = features[LABEL_COLUMN]
        fc0_w = tf.get_variable("fc0_w", [FLAGS.embedding_size * FLAGS.slot_nums + FLAGS.dense_nums, 400],
                                tf.float32,
                                initializer=tf.random_normal_initializer(stddev=1.0/tf.sqrt(tf.to_float(FLAGS.embedding_size * FLAGS.slot_nums + FLAGS.dense_nums))))
        fc0_b = tf.get_variable("fc0_b", [400], tf.float32,
                                initializer=tf.constant_initializer(value=0))
        fc1_w = tf.get_variable("fc1_w", [400, 400], tf.float32,
                                initializer=tf.random_normal_initializer(stddev=1.0/tf.sqrt(tf.to_float(400))))
        fc1_b = tf.get_variable("fc1_b", [400], tf.float32,
                                initializer=tf.constant_initializer(value=0))
        fc2_w = tf.get_variable("fc2_w", [400, 400], tf.float32,
                                initializer=tf.random_normal_initializer(stddev=1.0/tf.sqrt(tf.to_float(400))))
        fc2_b = tf.get_variable("fc2_b", [400], tf.float32,
                                initializer=tf.constant_initializer(value=0))
        fc3_w = tf.get_variable("fc3_w", [400, 2], tf.float32,
                                initializer=tf.random_normal_initializer(stddev=1.0/tf.sqrt(tf.to_float(400))))
        fc3_b = tf.get_variable("fc3_b", [2], tf.float32,
                                initializer=tf.constant_initializer(value=0))

        fc0 = tf.nn.relu(tf.matmul(concat, fc0_w) + fc0_b)
        fc1 = tf.nn.relu(tf.matmul(fc0, fc1_w) + fc1_b)
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
        fc_predict = tf.matmul(fc2, fc3_w) + fc3_b
        
        logits = fc_predict
        predict = tf.nn.softmax(logits=logits)
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_y, logits=logits)
        avg_cost = tf.reduce_mean(cost)
        positive_p = predict[:, 1]
        train_auc, train_update_op = tf.metrics.auc(
            labels=label_y,
            predictions=positive_p,
            name="auc")
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(avg_cost, global_step=global_step) 
        log_dir = FLAGS.model_dir + '/checkpoint/local/model.ckpt'
        saver = tf.train.Saver(max_to_keep=None)
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            try:
                batch_id = 0
                total_time = 0.0
                while True:
                    start_time = time.time()
                    _, auc_v, update_op_v, loss_v, step = sess.run([optimizer, train_auc, train_update_op, avg_cost, global_step])
                    end_time = time.time()
                    total_time += end_time - start_time
                    batch_id += 1
                    if step % 100 == 0:
                        print_log("step: %d, auc: %f, loss: %f, speed: %f secs/batch" % (step, auc_v, loss_v, total_time / float(batch_id)))
                    if step % 44000 == 0:
                        saver.save(sess, log_dir, global_step=step)
                    if batch_id >= 44000:
                        batch_id = 0
                        total_time = 0.0
            except tf.errors.OutOfRangeError:
                print("there are a total of %d batchs" % step)
   
if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
