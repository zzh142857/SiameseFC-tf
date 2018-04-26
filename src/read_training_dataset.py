# -*- coding: utf-8 -*-
__author__ = "Zhenghao Zhao"

import tensorflow as tf

import os
import os.path

#default image shape
resize_height = 700  # 存储图片高度
resize_width = 700  # 存储图片宽度
channel = 3


def decode_single_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'z_raw': tf.FixedLenFeature([], tf.string),
            'x_raw': tf.FixedLenFeature([], tf.string),
            'z_pos_x': tf.FixedLenFeature([], tf.float32),
            'z_pos_y': tf.FixedLenFeature([], tf.float32),
            'z_target_w': tf.FixedLenFeature([], tf.float32),
            'z_target_h': tf.FixedLenFeature([], tf.float32),
            'x_pos_x': tf.FixedLenFeature([], tf.float32),
            'x_pos_y': tf.FixedLenFeature([], tf.float32),
            'x_target_w': tf.FixedLenFeature([], tf.float32),
            'x_target_h': tf.FixedLenFeature([], tf.float32)
            
        }
    )

    
    #decode image of z and x
    z = tf.cast(tf.decode_raw(features['z_raw'], tf.uint8), tf.float64) #shape(h, w, c)    
    #print("shape of z:  ", z.get_shape().as_list())
    z = tf.reshape(z, [resize_width, resize_height, channel])
    x = tf.cast(tf.decode_raw(features['x_raw'], tf.uint8), tf.float64)
    x = tf.reshape(x, [resize_width, resize_height, channel])
    x = x * (2. / 255.) - 1
    z = z * (2. / 255.) - 1
    
    # coordinte
    z_pos_x = tf.cast(features['z_pos_x'] * resize_width, tf.int32)
    z_pos_y = tf.cast(features['z_pos_y'] * resize_width, tf.int32)
    z_target_w = tf.cast(features['z_target_w'] * resize_width, tf.int32)
    z_target_h = tf.cast(features['z_target_h'] * resize_width, tf.int32)
    x_pos_x = tf.cast(features['x_pos_x'] * resize_width, tf.int32)
    x_pos_y = tf.cast(features['x_pos_y'] * resize_width, tf.int32)
    x_target_w = tf.cast(features['x_target_w'] * resize_width, tf.int32)
    x_target_h = tf.cast(features['x_target_h'] * resize_width, tf.int32)

 
    return  z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h
    


def read_tfrecord(filename, num_epochs, batch_size, width = 700, height = 700):
    resize_width = width
    resize_height = height
    
    filename_queue = tf.train.string_input_producer([filename  + ".tfrecords" ] )
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h = decode_single_example(serialized_example)

    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 20
    capacity = min_after_dequeue + 3 * batch_size

    return tf.train.shuffle_batch(
    [z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    
    


    


