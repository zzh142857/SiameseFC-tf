
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

import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores


# gpu_device = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

# read default parameters and override with custom ones
def trainer(hp, run, design, small_score_sz, batch_size, z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h, image, templates_z, scores, loss, train_step):
    
    



    # run_metadata = tf.RunMetadata()
    # run_opts = {
    #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #     'run_metadata': run_metadata,
    # }

    run_opts = {}

    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        print("Session started......")
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        step = 0
        while (True):
            try:
                z_, x_, z_pos_x_, z_pos_y_, z_target_w_, z_target_h_, x_pos_x_, x_pos_y_, x_target_w_, x_target_h_= sess.run([z, x, z_pos_x, z_pos_y, z_target_w, z_target_h, x_pos_x, x_pos_y, x_target_w, x_target_h])
                #print(z_pos_x_, z_pos_y_, z_target_w_, z_target_h_, x_pos_x_, x_pos_y_, x_target_w_, x_target_h_)

                context = design.context*(z_target_w_+z_target_h_)
                z_sz = tf.cast(tf.sqrt(tf.constant(z_target_w_+context)*tf.constant(z_target_h_+context)), tf.float64)#(w +2p)*(h+2p)
                z_sz_ = sess.run(z_sz)
                print("z_sz_: ", z_sz_.shape)
                            
                templates_z_ = sess.run([templates_z], feed_dict={
                                                                                siam.batched_pos_x_ph: z_pos_x_,
                                                                                siam.batched_pos_y_ph: z_pos_y_,
                                                                                siam.batched_z_sz_ph: z_sz_,
                                                                                image: z_})
                
                
                
                
              
                label = _create_gt_label_by_rescaling(batch_size, small_score_sz, x_target_w_, x_target_h_, x_.shape[1], x_.shape[2])
                
                scores_, loss_, _= sess.run(
                    [scores, loss,  train_step],
                    feed_dict={
                       
                        siam.batched_z_sz_ph: z_sz_,
                        siam.batched_pos_x_ph: x_pos_x_,
                        siam.batched_pos_y_ph: x_pos_y_,
                        siam.batched_x_sz0_ph: z_sz_,
                        siam.batched_x_sz1_ph: z_sz_,
                        siam.batched_x_sz2_ph: z_sz_ * 1.025,
                        templates_z: np.squeeze(templates_z_),
                        image: x_,
                        siam.label: label
                    })

                #print("shape of s_scores", s_scores_)
                step += 1
                if step % 5 == 0:
                    print("step %d, loss=%f"%(step, loss_))
                       
                

            except tf.errors.OutOfRangeError:
                print("End of training")  # ==> "End of dataset"
                break

            finally:    
                # Finish off the filename queue coordinator.
                coord.request_stop()
                coord.join(threads)             


def _create_gt_label_by_rescaling(batch_size, small_score_sz, x_target_w_, x_target_h_, x_w, x_h):
    label = [[[0. for y_coor in range(small_score_sz)] for x_coor in range(small_score_sz)] for c in range(5)]
    for i in range(5):
        label_w = int(x_target_w_[i] * small_score_sz / x_w)
        label_h = int(x_target_h_[i] * small_score_sz / x_h)
        for x_index in range(label_h):
            for y_index in range(label_w):
                label[i][int(small_score_sz / 2 + y_index - label_w / 2)][int(small_score_sz / 2 + x_index - label_h / 2)] = 1.
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


