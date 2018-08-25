#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:40:59 2018

@author: wsw
"""

# make train dataset

import tensorflow as tf
import os
import time
slim = tf.contrib.slim

tf.reset_default_graph()

def parser_example(serialized_example):
    features = {'raw_image':tf.FixedLenFeature([],tf.string)}
    example = tf.parse_single_example(serialized_example,
                                      features=features)
    image = tf.decode_raw(example['raw_image'],out_type=tf.uint8)
    # if use tf.train.batch make batch data must specify shape
    image = tf.reshape(image,shape=[64,64,3])
    image = tf.cast(image,dtype=tf.float32)
    image = image/127.5-1.0
    return image


def make_train_dataset(epoch=200,batchsize=128,shuffle=True):
    
    tfrecords_path = os.path.join('./data','faces.tfrecords')
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(parser_example,num_parallel_calls=10)
    # skip is using to ignore final small batch in order to 
    # keep every batch have same shape
    dataset = dataset.shuffle(10000).repeat(epoch).skip(23).batch(batchsize)
    return dataset
    

def test_dataset():
    dataset = make_train_dataset(epoch=1)
    train_iter = dataset.make_one_shot_iterator()
    image_batch = train_iter.get_next()
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        try:
            start = time.time()
            step = 1
            while not coord.should_stop():
                img_batch = sess.run(image_batch)
                print('Step:%d'%step,img_batch.shape)
                step += 1
        except tf.errors.OutOfRangeError:
            end = time.time()
            print(end-start)
            import matplotlib.pyplot as plt
            import numpy as np
            image = img_batch[0]
            print(np.min(image))
            plt.imshow(image)
            plt.show()
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':       
    test_dataset()