# -*- coding: utf-8 -*-
__author__ = "Zhenghao Zhao"

import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from src.region_to_bbox import region_to_bbox


resize_height = 200  # 存储图片高度
resize_width = 200  # 存储图片宽度
original_height = 200
original_width = 200
channel = 1

def load_file(examples_list_file):
    lines = np.genfromtxt(examples_list_file, delimiter="\n", dtype=[('col1', 'S256')])
    examples = []
    for example in lines:
        examples.append(str(example, 'utf-8'))
    return np.asarray(examples), len(lines)


def extract_coord(filename):
    txt_filename = filename[:-4] + '.txt'
    # print (txt_filename)
    fp = open(txt_filename)
    coord_line = fp.readline()
    # print(coord_line)
    coord = coord_line.split(' ')
    return int(coord[0]), float(coord[1]), float(coord[2]), float(coord[3]), float(coord[4])


def transform2tfrecord(data_folder, name, output_directory):
    """
        Input:
            data_folder: relative path of folder who contains all the vedio folders for training.

    """
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
        
    #init tfrecord file
    filename = os.path.join(output_directory, name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
        
    cur_dir = os.getcwd()
    data_folder = os.path.join(cur_dir, data_folder)
    #get a list of dirs in data_folder, in each of which contains a training vedio
    vedio_folder_list = [dir for dir in os.listdir(data_folder) if not os.path.isfile(os.path.join(data_folder, dir))]

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
        z_raw = z.tostring()
        z_gt = gts[0].strip("\n").split(",")   
        assert len(z_gt) == 4
        z_pos_x, z_pos_y, z_target_w, z_target_h = region_to_bbox(z_gt)
        
        
        for _example in _examples[1: ]:

            assert len(_example) == 2
            gt = _example[1].strip("\n").split(",") 
            assert len(gt) == 4
            
            img_file = os.path.join(vedio_folder, _example[0])
            x_pos_x, x_pos_y, x_target_w, x_target_h = region_to_bbox(gt)                
            image = cv2.imread(img_file)
            
            image_raw = image.tostring()
 
            example = tf.train.Example(features=tf.train.Features(feature={
                'z_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[z_raw])),
                'x_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[z_raw])),
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
    x = tf.cast(tf.decode_raw(features['x_raw'], tf.uint8), tf.float32)
      
    # coordinte
    z_pos_x = tf.cast(features['z_pos_x'], tf.float32)
    z_pos_y = tf.cast(features['z_pos_y'], tf.float32)
    z_target_w = tf.cast(features['z_target_w'], tf.float32)
    z_target_h = tf.cast(features['z_target_h'], tf.float32)
    x_pos_x = tf.cast(features['x_pos_x'], tf.float32)
    x_pos_y = tf.cast(features['x_pos_y'], tf.float32)
    x_target_w = tf.cast(features['x_target_w'], tf.float32)
    x_target_h = tf.cast(features['x_target_h'], tf.float32)

    return z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h
    
def normalize(image, x, y, r):
  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = image * (1. / 255) - 0.5
  x = x / original_width
  y = y / original_height
  return image, x, y, r

def read_tfrecord(filename, num_epochs, batch_size):
    with tf.name_scope('input'):
    # TFRecordDataset opens a protobuf and reads entries line by line
    # could also be [list, of, filenames]
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(decode_single_example).repeat(num_epochs).map(normalize)
    
        #the parameter is the queue size
        dataset = dataset.shuffle(5 * batch_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        iterator = dataset.make_initializable_iterator()
    return dataset, iterator

def run_load(train_file, name, output_directory, height, width, num_epochs = 100, batch_size = 50):
    resize_height = height
    resize_width = width
    transform2tfrecord(train_file, name, output_directory)
    dataset, iterator = read_tfrecord(os.path.join(output_directory, name + '.tfrecords'), num_epochs, batch_size)

    return dataset, iterator

def test():
    train_file = os.path.join(os.getcwd(), "test_data", "train.txt")
    output_directory = os.path.join(os.getcwd(), "test_data")
    transform2tfrecord(train_file, 'train', output_directory, 200, 200)  # 转化函数
    img, x, y, r = read_tfrecord(os.path.join(output_directory, name + '.tfrecords')) #读取函数


if __name__ == '__main__':
    test()
