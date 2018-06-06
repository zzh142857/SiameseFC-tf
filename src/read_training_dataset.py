# -*- coding: utf-8 -*-
__author__ = "Zhenghao Zhao"

import tensorflow as tf

import os
import os.path


class myReader(object):
    def __init__(self, width, height, channel):
        self.resize_width = width
        self.resize_height = height
        self.channel = channel
        
    def decode_single_example(self, serialized_example):
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
        #print("shape of z:  ", z.get_shape())
        z = tf.reshape(z, [self.resize_width, self.resize_height, self.channel])
        x = tf.cast(tf.decode_raw(features['x_raw'], tf.uint8), tf.float64)
        x = tf.reshape(x, [self.resize_width, self.resize_height, self.channel])
        #normalize image
        x = x * (2. / 255.) - 1
        z = z * (2. / 255.) - 1
        
        # coordinte
        z_pos_x = tf.cast(features['z_pos_x'] * self.resize_width, tf.int32)
        z_pos_y = tf.cast(features['z_pos_y'] * self.resize_width, tf.int32)
        z_target_w = tf.cast(features['z_target_w'] * self.resize_width, tf.int32)
        z_target_h = tf.cast(features['z_target_h'] * self.resize_width, tf.int32)
        x_pos_x = tf.cast(features['x_pos_x'] * self.resize_width, tf.int32)
        x_pos_y = tf.cast(features['x_pos_y'] * self.resize_width, tf.int32)
        x_target_w = tf.cast(features['x_target_w'] * self.resize_width, tf.int32)
        x_target_h = tf.cast(features['x_target_h'] * self.resize_width, tf.int32)

     
        return  z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h
        


    def read_tfrecord(self, filename, num_epochs, batch_size):
        
        filename_queue = tf.train.string_input_producer([filename  + ".tfrecords" ], num_epochs = num_epochs)
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)
        z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h = self.decode_single_example(serialized_example)

        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        
        #due to limited memory of my graphic card, I set the size of queue to s small number
        min_after_dequeue = 20
        capacity = min_after_dequeue + 3 * batch_size

        return tf.train.shuffle_batch(
        [z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

        
    


    


