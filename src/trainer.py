
import tensorflow as tf
print('Using Tensorflow '+tf.__version__)
import matplotlib.pyplot as plt
import sys
# sys.path.append('../')
import os
import csv
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores
import cv2
#from run_tracker_evaluation_v2 import main

# gpu_device = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

# read default parameters and override with custom ones
def trainer(hp, run, design, final_score_sz, image, templates_z, scores, loss, train_step, distance_to_gt, batched_data, z_crops, x_crops, siamNet, summary, tz, max_pos_x, max_pos_y):
    
    # run_metadata = tf.RunMetadata()
    # run_opts = {
    #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #     'run_metadata': run_metadata,
    # }

    run_opts = {}
    z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h = batched_data
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    config = tf.ConfigProto()    
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=1000)
    with tf.Session(config = config) as sess:
        #saver.restore(sess, "output/saver5_vedio_unormalized_half_labelsize-4000")
        print("Session started......")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        step = 0
        #summary_writer = tf.summary.FileWriter('output', sess.graph)
        while (True):
            step += 1;
            try:
                z_, x_, z_pos_x_, z_pos_y_, z_target_w_, z_target_h_, x_pos_x_, x_pos_y_, x_target_w_, x_target_h_= sess.run([z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h])
                # print(z_pos_x_, z_pos_y_, z_target_w_, z_target_h_, x_pos_x_, x_pos_y_, x_target_w_, x_target_h_)

                """
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.circle(x_[1],(int(x_pos_x_[1]),int(x_pos_y_[1])),2,(255,0,0),-11)
                
                cv2.rectangle(x_[1], (int(x_pos_x_[1] - x_target_w_[1] / 2), int(x_pos_y_[1] - x_target_h_[1] / 2)), (int(x_pos_x_[1] + x_target_w_[1] / 2), int(x_pos_y_[1] + x_target_h_[1] / 2)), (255,0,0), 2)
                cv2.imshow('image',x_[1])
                cv2.waitKey(0)
                
                """              
                context = design.context*(z_target_w_+z_target_h_)
                z_sz = tf.cast(tf.sqrt(tf.constant(z_target_w_+context)*tf.constant(z_target_h_+context)), tf.float64)#(w +2p)*(h+2p)
                x_sz = float(design.search_sz) / design.exemplar_sz * z_sz
                z_sz_, x_sz_ = sess.run([z_sz, x_sz])
                #print("z_sz_: ", z_sz_.shape)
                            
                templates_z_, z_crops_ = sess.run([templates_z, z_crops], feed_dict={
                                                                                siamNet.batched_pos_x_ph: z_pos_x_,
                                                                                siamNet.batched_pos_y_ph: z_pos_y_,
                                                                                siamNet.batched_z_sz_ph: z_sz_,
                                                                                image: z_})
                """             
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.circle(z_crops_[0],(int(design.exemplar_sz / 2.),int(design.exemplar_sz / 2.)),2,(255,125,125),-11)
                label_w = int(z_target_w_[0] * design.exemplar_sz / z_sz_[0])
                label_h = int(z_target_h_[0] * design.exemplar_sz / z_sz_[0])
                cv2.rectangle(z_crops_[0], (int(design.exemplar_sz / 2. - label_w / 2), int(design.exemplar_sz / 2. - label_h / 2)), (int(design.exemplar_sz / 2. + label_w / 2), int(design.exemplar_sz / 2. + label_h / 2)), (255,0,0), 2)
      
                cv2.imshow('image',z_crops_[0])
                cv2.waitKey(0)
                return
                """
                
                label = _create_gt_label_final_score_sz(design.batch_size, final_score_sz, x_target_w_, x_target_h_, x_sz_, design.search_sz)
                
                
                scores_, loss_, _, x_crops_, summary_, d, max_pos_x_, max_pos_y_= sess.run(
                    [scores, loss,  train_step, x_crops, summary, distance_to_gt, max_pos_x, max_pos_y],
                    feed_dict={
                       
                        siamNet.batched_z_sz_ph: z_sz_,
                        siamNet.batched_pos_x_ph: x_pos_x_,
                        siamNet.batched_pos_y_ph: x_pos_y_,
                        siamNet.batched_x_sz0_ph: x_sz_,
                        siamNet.batched_x_sz1_ph: x_sz_,
                        siamNet.batched_x_sz2_ph: x_sz_ * 1.02,
                        templates_z: np.squeeze(templates_z_),
                        image: x_,
                        siamNet.label: label
                    })
                """
                if step % 2 == 0:
                    plt.subplot(121)
                    plt.imshow(np.squeeze(scores_[0] + 10)/20 , cmap = 'gray')
                    plt.subplot(122)
                    plt.imshow(x_crops_[0] + 0.5)
                    plt.show()
                    plt.pause(5)
                """
                #scores_ = np.squeeze(scores_)
                # find scale with highest peak (after penalty)
                #scale_id = np.argmax(scores_, axis=(1,2))
                #print(scale_id)
                """
                for channel in scores_:
                    for row in channel:
                        for num in row:
                            if num > 100:
                                print(num)
                """
                
                """
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                
                for xx in range(255):
                    for yy in range(255):
                        if label[0][xx][yy] > 0:
                            cv2.circle(x_crops_[0],(int(yy),int(xx)),1,(255,125,125),-11)
                
                label_w = int(x_target_w_[0] * design.search_sz / x_sz_[0])
                label_h = int(x_target_h_[0] * design.search_sz / x_sz_[0])
                cv2.rectangle(x_crops_[0], (int(- label_w / 2 + 257 / 2), int( - label_h / 2+ 257 / 2)), (int( + label_w / 2+ 257/2), int(+ label_h / 2+ 257/2)), (255,0,0), 2)
                cv2.imshow('image',x_crops_[0] + 0.5)
                
                cv2.waitKey(0)
                """
                
                
                
                
                
                #print("shape of s_scores", s_scores_)
                
                if step % 5 == 0:
                    print("step %d, loss=%f, distance_to_gt=%f"%(step, loss_, d))
                    """
                    summary_writer.add_summary(summary_, step)
                    summary_writer.flush()
                    """
                    
                if step % 500 == 0:
                    save_path = saver.save(sess, os.path.join(desing.saver_folder, design.path_ckpt) , global_step = step)
                    #main(step)
                
                    
                
                       
            

            except Exception as e:
                print(e)  # ==> "End of dataset"
                break
                
           
            # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)             


def _create_gt_label_final_score_sz(batch_size, final_score_sz, x_target_w_, x_target_h_, x_sz, search_sz):
    label = [[[-1. for y_coor in range(final_score_sz)] for x_coor in range(final_score_sz)] for c in range(batch_size)]
    for i in range(batch_size):
        label_w = int(x_target_w_[i] * search_sz / x_sz[i])   // 2
        label_h = int(x_target_h_[i] * search_sz / x_sz[i]) // 2
        for x_index in range(label_w):
            for y_index in range(label_h):
                label[i][int(final_score_sz / 2. + y_index - label_h / 2.)][int(final_score_sz / 2. + x_index - label_w / 2.)] = 1.

    return label

def _create_gt_label_final_score_sz_circle(batch_size, final_score_sz, x_target_w_, x_target_h_, x_sz, search_sz):
    label = [[[-1. for y_coor in range(final_score_sz)] for x_coor in range(final_score_sz)] for c in range(batch_size)]

    for i in range(batch_size):
        label_w = int(x_target_w_[i] * search_sz / x_sz[i])  
        label_h = int(x_target_h_[i] * search_sz / x_sz[i]) 
        for x_index in range(label_w):
            for y_index in range(label_h):
                label[i][int(final_score_sz / 2. + y_index - label_h / 2.)][int(final_score_sz / 2. + x_index - label_w / 2.)] = 1.

    return label

def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


