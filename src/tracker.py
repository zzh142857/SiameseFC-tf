import tensorflow as tf
print('Using Tensorflow '+tf.__version__)
import matplotlib.pyplot as plt
import sys
import os
import csv
import numpy as np
from PIL import Image
import time
import cv2
import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores


# read default parameters and override with custom ones

def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, 
    final_score_sz, image, templates_z, scores, path_ckpt, siamNet):
    """
        run the tracking steps under tensorflow session.
        
        Inputs:
            hp, run, design: system parameters.
            
            frame_name_list: a list of paths for all frames in the tracking vedio.
            
            pos_x, pos_y, target_w, target_h: target position and size in the 
                first frame from ground thruth, will be updated during tracking.
            
            final_score_sz: size of the final score map after bilinear interpolation.
            
            image, templates_z, scores: tensors that will be run in tensorflow session.
                See siamese.py for detailed explanation.
                
            path_ckpt: path of the checkpoint file used to retore model variables.
            
            siamNet: an instance of siamese network class.
            
        Returns:
            bboxes: a list of the predicted bboxes
            
            speed: average tracking speed(fps)
    """
    num_frames = np.size(frame_name_list)
    # stores tracker's output for evaluation
    bboxes = np.zeros((num_frames,4))

    scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    # cosine window to penalize large displacements    
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    context = design.context*(target_w+target_h)
    z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))#(w +2p)*(h+2p)
    x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

    # thresholds to saturate patches shrinking/growing
    min_z = hp.scale_min * z_sz
    max_z = hp.scale_max * z_sz
    min_x = hp.scale_min * x_sz
    max_x = hp.scale_max * x_sz

    # run_metadata = tf.RunMetadata()
    # run_opts = {
    #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #     'run_metadata': run_metadata,
    # }

    run_opts = {}
    saver = tf.train.Saver() 
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        saver.restore(sess, path_ckpt)
        print("Model restored from: ", path_ckpt)
        print("Start tracking......")
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # save first frame position (from ground-truth)
        bboxes[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h  

        # Read the first frame as z, and input into conv net to get its feature map
        z_image = cv2.imread(frame_name_list[0])      
        image_, templates_z_ = sess.run([image, templates_z], feed_dict={
                                                                        siamNet.batched_pos_x_ph: [pos_x],
                                                                        siamNet.batched_pos_y_ph: [pos_y],
                                                                        siamNet.batched_z_sz_ph: [z_sz],
                                                                        image: [z_image / 255.  - 0.5]})
        new_templates_z_ = templates_z_

        t_start = time.time()
        
        # Get an image from the queue
        for i in range(1, num_frames):        
            scaled_exemplar = z_sz * scale_factors
            scaled_search_area = x_sz * scale_factors
            scaled_target_w = target_w * scale_factors
            scaled_target_h = target_h * scale_factors
            
            # Read the next frame as x, input x into conv net, with the featre 
            # map of z to get the final score map
            x_image = cv2.imread(frame_name_list[i])
            image_, scores_ = sess.run(
                [image, scores],
                feed_dict={
                    siamNet.batched_pos_x_ph: [pos_x],
                    siamNet.batched_pos_y_ph: [pos_y],
                    siamNet.batched_x_sz0_ph: [scaled_search_area[0]],
                    siamNet.batched_x_sz1_ph: [scaled_search_area[1]],
                    siamNet.batched_x_sz2_ph: [scaled_search_area[2]],
                    templates_z: np.squeeze(templates_z_),
                    image: [x_image / 255. - 0.5],
                }, **run_opts)
            
            # visualize the output score map
            """
            plt.imshow(np.squeeze(scores_[0]), cmap = 'gray')
            plt.show()
            plt.pause(5)
            """
            
            #finalize the score map
            scores_ = np.squeeze(scores_)
            # penalize change of scale
            scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
            scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
            # find scale with highest peak (after penalty)
            new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
            # update scaled sizes
            x_sz = (1-hp.scale_lr)*x_sz + hp.scale_lr*scaled_search_area[new_scale_id]        
            target_w = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
            target_h = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
            # select response with new_scale_id
            score_ = scores_[new_scale_id,:,:]
            score_ = score_ - np.min(score_)
            score_ = score_/np.sum(score_)
            # apply displacement penalty
            score_ = (1-hp.window_influence)*score_ + hp.window_influence*penalty
            
            # visualize the finalized score map
            """
            plt.imshow(np.squeeze(score_), cmap = 'gray')
            plt.show()
            plt.pause(5)
            """
            
            pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
            # convert <cx,cy,w,h> to <x,y,w,h> and save output
            bboxes[i,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
            # update the target representation with a rolling average
           
            if hp.z_lr>0:
                new_templates_z_ = sess.run([templates_z], feed_dict={
                                                                siamNet.batched_pos_x_ph: [pos_x],
                                                                siamNet.batched_pos_y_ph: [pos_y],
                                                                siamNet.batched_z_sz_ph: [z_sz],
                                                                image: image_
                                                                })

                templates_z_=(1-hp.z_lr)*np.asarray(templates_z_) + hp.z_lr*np.asarray(new_templates_z_)
            
            # update template patch size
            z_sz = (1-hp.scale_lr)*z_sz + hp.scale_lr*scaled_exemplar[new_scale_id]
            
            if run.visualization:
                show_frame((image_[0] + 0.5) * 255 , bboxes[i,:], 1)        
        
        t_elapsed = time.time() - t_start
        speed = num_frames/t_elapsed

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads) 

        # from tensorflow.python.client import timeline
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # trace_file = open('timeline-search.ctf.json', 'w')
        # trace_file.write(trace.generate_chrome_trace_format())

    plt.close('all')
    
    return bboxes, speed

def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    #print(p)
    
    tmp1 = score[p]
    score[p] = -float('inf')
    p_2 = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    tmp2 = score[p_2]
    score[p_2] = -float('inf')
    p_3 = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    #score[p] = tmp1
    #score[p_2] = tmp2
    p = (p + p_2 + p_3 ) / 3
    
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop  *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


