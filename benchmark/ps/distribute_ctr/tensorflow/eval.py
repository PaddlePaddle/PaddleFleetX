#!/usr/bin/env python
# coding=utf-8

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import time
import argparse
import shutil
import sys
import os
import glob
import json
import random
from datetime import date, timedelta

# import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import re
import data_generator as reader

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("embedding_size", 10, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 1000, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_integer("dict_size", 1000001, "dict_size")
tf.app.flags.DEFINE_integer("dense_nums", 13, "dense feature num")
tf.app.flags.DEFINE_integer("slot_nums", 26, "sparse feature num")

tf.app.flags.DEFINE_string("test_data_dir", 'test_data', "test data dir")
tf.app.flags.DEFINE_string("task_mode", '', "task_mode")
tf.app.flags.DEFINE_string("checkpoint_path", '', "directory to save checkpoint file")
tf.app.flags.DEFINE_string("result_path", '', "directory to save evaluate result")

def get_file_list():
    data_dir = FLAGS.test_data_dir
    data_files = os.listdir(data_dir)
    file_list = list()
    for data_file in data_files:
        file_list.append(data_dir + '/' + data_file)
    print("File list:" + str(file_list))
    return file_list

def get_batch(reader, batch_size):
    example_batch = []
    for _ in range(FLAGS.slot_nums + 2):
        example_batch.append([])
    for example in reader():
        for i in range(len(example)):
            example_batch[i].append(example[i])
        if len(example_batch[0]) >= batch_size:
            yield example_batch
            for _ in range(FLAGS.slot_nums + 2):
                example_batch[_] = []

def model(words):
    fc0_w = tf.get_variable("fc0_w", [FLAGS.slot_nums * FLAGS.embedding_size + FLAGS.dense_nums, 400], tf.float32)
    fc0_b = tf.get_variable("fc0_b", [400], tf.float32)
    fc1_w = tf.get_variable("fc1_w", [400, 400], tf.float32)
    fc1_b = tf.get_variable("fc1_b", [400], tf.float32)
    fc2_w = tf.get_variable("fc2_w", [400, 400], tf.float32)
    fc2_b = tf.get_variable("fc2_b", [400], tf.float32)
    fc3_w = tf.get_variable("fc3_w", [400, 2], tf.float32)
    fc3_b = tf.get_variable("fc3_b", [2], tf.float32)
    embeddings = tf.get_variable("emb", [FLAGS.dict_size, FLAGS.embedding_size], tf.float32)
    def embedding_layer(input_):
        return tf.reduce_sum(tf.nn.embedding_lookup(embeddings, input_), axis=1)

    sparse_embed_seq = list(map(embedding_layer, words[1:-1]))
    concat = tf.concat(sparse_embed_seq + words[0:1], axis=1)
    label_y = words[-1]

    fc0 = tf.nn.relu(tf.matmul(concat, fc0_w) + fc0_b)
    fc1 = tf.nn.relu(tf.matmul(fc0, fc1_w) + fc1_b)
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
    fc_predict = tf.matmul(fc2, fc3_w) + fc3_b
    logits = fc_predict
    predict = tf.nn.softmax(logits=logits)
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(label_y), logits=logits)
    avg_cost = tf.reduce_mean(cost)
    positive_p = predict[:, 1]
    auc, update_op = tf.metrics.auc(
        labels=label_y,
        predictions=positive_p, name='auc')
    return auc, update_op, avg_cost

def main(result_file_path, checkpoint_path):
    result = dict()
    if os.path.exists(result_file_path):
        with open(result_file_path) as f:
            result = json.load(f)

    file_list = get_file_list()
    print("there are a total of %d test files" % (len(file_list)))
    print(file_list)
    test_generator = reader.CriteoDataset(FLAGS.dict_size) 
    dense_input = tf.placeholder(tf.float32, [None, FLAGS.dense_nums], name="dense_input")
    sparse_input = [tf.placeholder(tf.int64, [None, 1], name="C" + str(i)) for i in range(1, 27)]
    label_y = tf.placeholder(tf.int64, [None, 1], name="label")
    words = [dense_input] + sparse_input + [label_y]
    auc, update_op, avg_cost = model(words)
    while True:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt=tf.train.get_checkpoint_state(checkpoint_path)
            if ckpt and ckpt.all_model_checkpoint_paths:
                for path in ckpt.all_model_checkpoint_paths:
                    sess.run(tf.local_variables_initializer())
                    global_step=path.split('/')[-1].split('-')[-1]
                    if global_step in result:
                        continue
                    if not os.path.exists(path+'.index'):
                        print("checkpoint file: %s has expired!" % path)
                        continue
                    print("Start to inference ==> %s" % (path))
                    saver.restore(sess,path)
                    local_step = 0
                    auc_val = 0.0
                    for words_input in get_batch(test_generator.test(file_list), FLAGS.batch_size):
                        feed_dict = {}
                        for i, item in enumerate(words):
                            feed_dict[item] = words_input[i]
                        auc_val, _, avg_cost_val = sess.run([auc, update_op, avg_cost], feed_dict=feed_dict)
                        if local_step % 100 == 0:
                            print("global step: %s, test batch step: %d, test auc: %f, test loss: %f" % (global_step, local_step, auc_val, avg_cost_val))
                        local_step += 1
                    result[global_step] = str(auc_val)
                    with open(result_file_path, 'w') as f:
                        json.dump(result, f)
            else:
                print('No checkpoint file found')
        print("sleeping 300s")
        time.sleep(300)
if __name__ == '__main__':
    print("task_mode: %s" % FLAGS.task_mode)
    print("checkpoint path: %s" % FLAGS.checkpoint_path)
    print("result path is: %s" % FLAGS.result_path)
    print("evaluate %s" % FLAGS.checkpoint_path)
    result_file_path = os.path.join(os.path.abspath(FLAGS.result_path), FLAGS.task_mode) + '.json'
    result = dict()
    if os.path.exists(result_file_path):
        with open(result_file_path) as f:
            result = json.load(f)
    main(result_file_path, FLAGS.checkpoint_path)


