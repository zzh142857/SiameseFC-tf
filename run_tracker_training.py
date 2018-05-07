from __future__ import division
import sys
import os
import src.siamese as siam
from src.trainer import trainer
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
import src.read_training_dataset 


"""
    training procedure:
    1,input z, x, pos_x, pos_y, w, d and gt of x
    2,pad and crop z,x, generate only one version
    3,calculate score map
    4,calculate loss
    5,bp, update variable
"""


def main():
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    
    # build the computational graph of Siamese fully-convolutional network
    siamNet = siam.Siamese(design.batch_size)
    # get tensors that will be used during training
    image, z_crops, x_crops, templates_z, scores, loss, train_step, distance_to_gt, summary= siamNet.build_tracking_graph_train(final_score_sz, design, env, hp)
 
    # read tfrecodfile holding all the training data
    data_reader = src.read_training_dataset.myReader(design.resize_width, design.resize_height, design.channel)
    batched_data = data_reader.read_tfrecord(os.path.join(env.tfrecord_path, env.tfrecord_filename), num_epochs = design.num_epochs, batch_size = design.batch_size)
    
    # run trainer
    trainer(hp, run, design, final_score_sz, batched_data, image, templates_z, scores, loss, train_step, distance_to_gt,  z_crops, x_crops, siamNet, summary)




if __name__ == '__main__':
    sys.exit(main())

