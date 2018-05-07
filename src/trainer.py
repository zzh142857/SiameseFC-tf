
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
def trainer(hp, run, design, final_score_sz, batched_data, image, templates_z, scores, loss, train_step, distance_to_gt, z_crops, x_crops, siamNet, summary):
    """
        run the training steps under tensorflow session.
        
        Inputs:
            hp, run, design: system parameters.
            
            final_score_sz: size of the final score map after bilinear interpolation.
            
            image, templates_z, scores, loss, train_step, distance_to_gt, z_crops, 
            x_crops: tensors that will be run in tensorflow session. See siamese.py 
                     for detailed explanation.
           
            siamNet: an instance of siamese network class.
            
        Returns:
       
    """
    
    # run_metadata = tf.RunMetadata()
    # run_opts = {
    #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #     'run_metadata': run_metadata,
    # }

    run_opts = {}
    
    # unpack data tensor from tfrecord
    z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h = batched_data
    
    # create saver to log check point
    saver = tf.train.Saver(max_to_keep=1000)
    # start a tf session with certain config
    config = tf.ConfigProto()    
    config.gpu_options.allow_growth = True    
    with tf.Session(config = config) as sess:
        #saver.restore(sess, "output/saver-1000")
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
                ## TODO: put all steps behind tensorflow
                # get real data from tfrecord first
                z_, x_, z_pos_x_, z_pos_y_, z_target_w_, z_target_h_, x_pos_x_, x_pos_y_, x_target_w_, x_target_h_= sess.run([z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h])
                
                # calculate crop size for z, x
                context = design.context*(z_target_w_+z_target_h_)
                z_sz = tf.cast(tf.sqrt(tf.constant(z_target_w_+context)*tf.constant(z_target_h_+context)), tf.float64)#(w +2p)*(h+2p)
                x_sz = float(design.search_sz) / design.exemplar_sz * z_sz
                z_sz_, x_sz_ = sess.run([z_sz, x_sz])
                
                # input z into conv net to get its feature map        
                templates_z_, z_crops_ = sess.run([templates_z, z_crops], feed_dict={
                                                                                siamNet.batched_pos_x_ph: z_pos_x_,
                                                                                siamNet.batched_pos_y_ph: z_pos_y_,
                                                                                siamNet.batched_z_sz_ph: z_sz_,
                                                                                image: z_})
                # visualize croped z image
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
                
                # create ground truth response map
                label = _create_gt_label_final_score_sz(design.batch_size, final_score_sz, x_target_w_, x_target_h_, x_sz_, design.search_sz)
                
                # input x into net, get x feature map, calculate score map, and its loss, iterate one train step
                scores_, loss_, _, x_crops_, summary_, distance_to_gt_ = sess.run(
                    [scores, loss,  train_step, x_crops, summary, distance_to_gt],
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
                
                # visualize the output score map
                """
                if step % 2 == 0:
                    plt.subplot(121)
                    plt.imshow(np.squeeze(scores_[0] + 10)/20 , cmap = 'gray')
                    plt.subplot(122)
                    plt.imshow(x_crops_[0] + 0.5)
                    plt.show()
                    plt.pause(5)
                """


                # visualize croped x image
                """
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)              
                label_w = int(x_target_w_[0] * design.search_sz / x_sz_[0])
                label_h = int(x_target_h_[0] * design.search_sz / x_sz_[0])
                cv2.rectangle(x_crops_[0], (int(- label_w / 2 + 257 / 2), int( - label_h / 2+ 257 / 2)), (int( + label_w / 2+ 257/2), int(+ label_h / 2+ 257/2)), (255,0,0), 2)
                cv2.imshow('image',x_crops_[0] + 0.5)                
                cv2.waitKey(0)
                """
                
                if step % 5 == 0:
                    print("step %d, loss=%f, distance_to_gt=%f"%(step, loss_, distance_to_gt_))
                    if run.write_summary:
                        summary_writer.add_summary(summary_, step)
                        summary_writer.flush()
                                        
                if step % 500 == 0:
                    save_path = saver.save(sess, os.path.join(design.saver_folder, design.path_ckpt) , global_step = step)
                    #main(step)

            except tf.errors.OutOfRangeError:
                print("End of training")  # ==> "End of dataset"
                break
                           
            # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)             


def _create_gt_label_final_score_sz(batch_size, final_score_sz, x_target_w_, x_target_h_, x_sz, search_sz):
    label = [[[-1. for y_coor in range(final_score_sz)] for x_coor in range(final_score_sz)] for c in range(batch_size)]
    for i in range(batch_size):
        label_w = int(x_target_w_[i] * search_sz / x_sz[i])
        label_h = int(x_target_h_[i] * search_sz / x_sz[i])
        for x_index in range(label_w):
            for y_index in range(label_h):
                label[i][int(final_score_sz / 2. + y_index - label_h / 2.)][int(final_score_sz / 2. + x_index - label_w / 2.)] = 1.

    return label

def _create_gt_label_final_half_score_sz(batch_size, final_score_sz, x_target_w_, x_target_h_, x_sz, search_sz):
    label = [[[-1. for y_coor in range(final_score_sz)] for x_coor in range(final_score_sz)] for c in range(batch_size)]

    for i in range(batch_size):
        label_w = int(x_target_w_[i] * search_sz / x_sz[i])     // 2
        label_h = int(x_target_h_[i] * search_sz / x_sz[i])  // 2
        for x_index in range(label_w):
            for y_index in range(label_h):
                label[i][int(final_score_sz / 2. + y_index - label_h / 2.)][int(final_score_sz / 2. + x_index - label_w / 2.)] = 1.

    return label

