# -*- coding: utf-8 -*-
__author__ = "Zhenghao Zhao"

import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from src.region_to_bbox import region_to_bbox_normalized



def transform2tfrecord(data_folder, tfrecord_name, output_directory, resize_width, resize_height):
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
    data_folder = os.path.join(cur_dir, data_folder)
    #get a list of dirs in data_folder, in each of which contains a training vedio
    vedio_folder_list = sorted([dir for dir in os.listdir(data_folder) if not os.path.isfile(os.path.join(data_folder, dir))])[:1]

    for vedio_folder in vedio_folder_list:
        vedio_folder = os.path.join(data_folder, vedio_folder)
        #get a list of dirs in data_folder, in each of which contains a training vedio
        file_list = [dir for dir in os.listdir(vedio_folder) if os.path.isfile(os.path.join(vedio_folder, dir))]
        img_list = sorted([file for file in file_list if file.endswith(".jpg")])
        gt_file_name = "groundtruth.txt"
        assert os.path.exists(os.path.join(vedio_folder, gt_file_name))
        
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
            """
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image',z)
            cv2.waitKey(0)
            break
            """


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



    
if __name__ == "__main__":
    transform2tfrecord("data", "train_test", "output", resize_width = 700, resize_height = 700)




