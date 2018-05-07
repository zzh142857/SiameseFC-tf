# -*- coding: utf-8 -*-
__author__ = "Zhenghao Zhao"


import cv2
import os
import os.path
from src.region_to_bbox import region_to_bbox_normalized
from random import shuffle


def prepare_shuffled_list(data_folder, output_filename, output_directory, num_vedio):
    """
        Input:
            data_folder: relative path of folder who contains all the vedio folders for training.
            tfrecord_name: nameof the tfrecord file
            output_directory: relative dir which will contain the tfrecord file

    """
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
        
    #init tfrecord file
    
        
    cur_dir = os.getcwd()
    data_folder = os.path.join(cur_dir, data_folder, num_vedio)
    #get a list of dirs in data_folder, in each of which contains a training vedio
    vedio_folder_list = sorted([dir for dir in os.listdir(data_folder) if not os.path.isfile(os.path.join(data_folder, dir))])[:num_vedio]
    vedio_index = 0
    output_list = []
    for vedio_folder in vedio_folder_list:
        vedio_index += 1
        
        vedio_folder = os.path.join(data_folder, vedio_folder)
        #get a list of dirs in data_folder, in each of which contains a training vedio
        file_list = [dir for dir in os.listdir(vedio_folder) if os.path.isfile(os.path.join(vedio_folder, dir))]
        img_list = sorted([file for file in file_list if file.endswith(".jpg")])
        #print(img_list)
        gt_file_name = "groundtruth.txt"
        assert os.path.exists(os.path.join(vedio_folder, gt_file_name))
        
        gt_file = open(os.path.join(vedio_folder, gt_file_name), 'r')
        gts = gt_file.readlines()
        assert len(gts) == len(img_list)
        _examples = list(zip(img_list, gts))
        
        #prepare examplar z
        z = os.path.join(vedio_folder, img_list[0])
        z_img = cv2.imread(z)
        z_gt = gts[0].strip("\n").split(",")   
        assert len(z_gt) == 4
        z_pos_x, z_pos_y, z_target_w, z_target_h = region_to_bbox_normalized(z_gt, z_img.shape[1], z_img.shape[0])
        
        
        
        
        for _example in _examples[1: ]:

            assert len(_example) == 2
            gt = _example[1].strip("\n").split(",") 
            assert len(gt) == 4
            
            x = os.path.join(vedio_folder, _example[0])
            x_img = cv2.imread(x)
                           
                      
            x_pos_x, x_pos_y, x_target_w, x_target_h = region_to_bbox_normalized(gt, x_img.shape[1], x_img.shape[0]) 
            



            
            output_list.append(z + " " + x + " " + str(z_pos_x)+ " "+ str(z_pos_y)+ " "+  str(z_target_w)+ " "+ str(z_target_h)+ " "+  str(x_pos_x)+ " "+ str(x_pos_y)+ " "+  str(x_target_w)+ " "+  str(x_target_h))
            z = x
            z_pos_x, z_pos_y, z_target_w, z_target_h = x_pos_x, x_pos_y, x_target_w, x_target_h
    
    shuffle(output_list);
    with open(os.path.join(output_directory,output_filename) + ".txt","w+") as f:
        for output_file in output_list:
            f.write(output_file + "\n")   
        

    
if __name__ == "__main__":
    prepare_shuffled_list("vedio", "shuffled_data_list", "tfrecords", num_vedio = 78)




