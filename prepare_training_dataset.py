# -*- coding: utf-8 -*-
__author__ = "Zhenghao Zhao"

import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from src.region_to_bbox import region_to_bbox_normalized



def transform2tfrecord(data_file, tfrecord_name, output_directory, resize_width, resize_height):
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
    writer = tf.python_io.TFRecordWriter(filename)
        
    cur_dir = os.getcwd()
    data_folder = os.path.join(cur_dir, data_file)
    with open(data_folder, "r") as f:
        data_list = f.readlines()
        for data in data_list:
            z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h = data.strip("\n ").split(" ")
            z_pos_x = float(z_pos_x)
            z_pos_y = float(z_pos_y)
            z_target_w = float(z_target_w)
            z_target_h = float(z_target_h)
            x_pos_x = float(x_pos_x)
            x_pos_y = float(x_pos_y)
            x_target_w = float(x_target_w)
            x_target_h = float(x_target_h)
            z_img = cv2.imread(z)
            #z_img = cv2.resize(z_img, (resize_width,resize_height))
            x_img = cv2.imread(x)
            #x_img = cv2.resize(x_img, (resize_width,resize_height))
            z_raw = z_img.tostring()
            x_raw = x_img.tostring()
            

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
        
        
 
        
    writer.close()
    print(tfrecord_name + '.tfrecords'+" is written to "+output_directory)



    
if __name__ == "__main__":
    transform2tfrecord("output/train_5_vedio.txt", "train_5_vedio", "output", resize_width = 700, resize_height = 700)




