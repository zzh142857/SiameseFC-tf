from __future__ import division
import sys
import os
import src.siamese as siam
from src.trainer import trainer
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
from src.read_training_dataset import read_tfrecord


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
    # build TF graph once for all
    #filename, image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env)

    siamNet = siam.Siamese(design.batch_size)
    image, z_crops, x_crops, templates_z, scores, loss, train_step, distance_to_gt, summary, tz , max_pos_x, max_pos_y= siamNet.build_tracking_graph_train(final_score_sz, design, env, hp)
 
    batched_data = read_tfrecord(os.path.join(env.tfrecord_path, env.tfrecord_filename), num_epochs = design.num_epochs, batch_size = design.batch_size)
    

    trainer(hp, run, design, final_score_sz, image, templates_z, scores, loss, train_step, distance_to_gt, batched_data, z_crops, x_crops, siamNet, summary, tz, max_pos_x, max_pos_y)




if __name__ == '__main__':
    sys.exit(main())

