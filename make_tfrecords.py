#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:43:35 2018

@author: ws
"""

# make tfrecords

import tensorflow as tf
import os
import skimage.io as imageio
from scipy import misc
import glob


def get_tfrecords_example(imgpath):
    image = imageio.imread(imgpath)
    image = misc.imresize(image,size=[64,64])
    feature = {}
    feature['raw_image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    example = example.SerializeToString()
    return example

def make_tfrecords():
    imgdir = './faces/*.jpg'
    imgpaths = glob.glob(imgdir)
    nums = len(imgpaths)
    if not os.path.exists('./data'):
        os.makedirs('./data')
    writer = tf.python_io.TFRecordWriter('./data/faces.tfrecords')
    for i in range(nums):
        # get tfrecords example
        example = get_tfrecords_example(imgpaths[i])
        writer.write(example)
        visualize_bar(i,nums-1)
    writer.close()


def visualize_bar(step,nums):
    ratio = int(step/nums*100)
    rate = int(40*(step/nums))
    fmt = '\r[%s%s]-%d%% %d/%d'%(rate*'>',(40-rate)*'-',ratio,step,nums)
    print(fmt,end='',flush=True)


if __name__ == '__main__':
    make_tfrecords()