# -*- coding: utf-8 -*-
__author__ = "Zhenghao Zhao"

import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from src.region_to_bbox import region_to_bbox_normalized

#default image shape
resize_height = 700  # 存储图片高度
resize_width = 700  # 存储图片宽度
channel = 3



def transform2tfrecord(data_folder, tfrecord_name, output_directory):
    """
        Input:
            data_folder: relative path of folder who contains all the vedio folders for training.
            tfrecord_name: nameof the tfrecord file
            output_directory: relative dir which will contain the tfrecord file

    """
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
        
    #init tfrecord file
    filename = os.path.join(output_directory, tfrecord_name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename, options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
        
    cur_dir = os.getcwd()
    data_folder = os.path.join(cur_dir, data_folder)
    #get a list of dirs in data_folder, in each of which contains a training vedio
    vedio_folder_list = [dir for dir in os.listdir(data_folder) if not os.path.isfile(os.path.join(data_folder, dir))][:4]

    for vedio_folder in vedio_folder_list:
        vedio_folder = os.path.join(data_folder, vedio_folder)
        #get a list of dirs in data_folder, in each of which contains a training vedio
        img_list = [dir for dir in os.listdir(vedio_folder) if os.path.isfile(os.path.join(vedio_folder, dir))]
        gt_file_name = img_list.pop()
        img_list = img_list[: -2]
        assert gt_file_name.endswith(".txt")
        gt_file = open(os.path.join(vedio_folder, gt_file_name), 'r')
        gts = gt_file.readlines()
        assert len(gts) == len(img_list)
        _examples = list(zip(img_list, gts))
        
        #prepare examplar z
        z = cv2.imread(os.path.join(vedio_folder, img_list[0]))
        
        z_gt = gts[0].strip("\n").split(",")   
        assert len(z_gt) == 4
        z_pos_x, z_pos_y, z_target_w, z_target_h = region_to_bbox_normalized(z_gt, z.shape[1], z.shape[0])
        z = cv2.resize(z, (resize_width,resize_height))
        z_raw = z.tostring()
        
        
        for _example in _examples[1: ]:

            assert len(_example) == 2
            gt = _example[1].strip("\n").split(",") 
            assert len(gt) == 4
            
            img_file = os.path.join(vedio_folder, _example[0])
                           
            x = cv2.imread(img_file)            
            x_pos_x, x_pos_y, x_target_w, x_target_h = region_to_bbox_normalized(gt, x.shape[1], x.shape[0]) 
            x = cv2.resize(x, (resize_width,resize_height))
            x_raw = x.tostring()
            #print(z_pos_x, z_pos_y, z_target_w, z_target_h)
 
            example = tf.train.Example(features=tf.train.Features(feature={
                'z_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[z_raw])),
                'x_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_raw])),
                'z_pos_x': tf.train.Feature(float_list=tf.train.FloatList(value=[z_pos_x])),
                'z_pos_y': tf.train.Feature(float_list=tf.train.FloatList(value=[z_pos_y])),
                'z_target_w': tf.train.Feature(float_list=tf.train.FloatList(value=[z_target_w])),
                'z_target_h': tf.train.Feature(float_list=tf.train.FloatList(value=[z_target_h])),
                'x_pos_x': tf.train.Feature(float_list=tf.train.FloatList(value=[x_pos_x])),
                'x_pos_y': tf.train.Feature(float_list=tf.train.FloatList(value=[x_pos_y])),
                'x_target_w': tf.train.Feature(float_list=tf.train.FloatList(value=[x_target_w])),
                'x_target_h': tf.train.Feature(float_list=tf.train.FloatList(value=[x_target_h]))
                
            }))
            writer.write(example.SerializeToString())
            z_raw = x_raw
            z_pos_x, z_pos_y, z_target_w, z_target_h = x_pos_x, x_pos_y, x_target_w, x_target_h
            
        
    writer.close()
    print(tfrecord_name + '.tfrecords'+" is written to "+output_directory)
    
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
    z = tf.cast(tf.decode_raw(features['z_raw'], tf.uint8), tf.float32) #shape(h, w, c)    
    z = tf.reshape(z, [resize_width, resize_height, channel])
    x = tf.cast(tf.decode_raw(features['x_raw'], tf.uint8), tf.float32)
    x = tf.reshape(x, [resize_width, resize_height, channel])
    x = x * (1. / 255) - 0.5
    z = z * (1. / 255) - 0.5
    
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
    


def read_tfrecord(filename, num_epochs, batch_size):
    
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader(options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))

    _, serialized_example = reader.read(filename_queue)
    z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h = decode_single_example(serialized_example)

    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10 * batch_size
    capacity = min_after_dequeue + 3 * batch_size

    return tf.train.shuffle_batch(
    [z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h], batch_size=batch_size, capacity=capacity,
    min_after_dequeue=min_after_dequeue)

    
    
    


def run_load(train_file, name, output_directory, height, width, num_epochs = 20, batch_size = 20):
    resize_height = height
    resize_width = width
    #transform2tfrecord(train_file, name, output_directory)
    return read_tfrecord(os.path.join(output_directory, name + '.tfrecords'), num_epochs, batch_size)

    


