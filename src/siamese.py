import tensorflow as tf
import numpy as np
import scipy.io
import sys
import os.path
from src.convolutional import set_convolutional, set_convolutional_train
from src.crops import extract_crops_z, extract_crops_x, pad_frame, resize_images
sys.path.append('../')


# the follow parameters *have to* reflect the design of the network to be imported
_conv_stride = np.array([2,1,1,1,1])
_filtergroup_yn = np.array([0,1,0,1,1], dtype=bool)
_bnorm_yn = np.array([1,1,1,1,0], dtype=bool)
_relu_yn = np.array([1,1,1,1,0], dtype=bool)
_pool_stride = np.array([2,1,0,0,0]) # 0 means no pool
_pool_sz = 3
_bnorm_adjust = True
assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), ('These arrays of flags should have same length')
assert all(_conv_stride) >= True, ('The number of conv layers is assumed to define the depth of the network')
_num_layers = len(_conv_stride)
hann_1d = np.expand_dims(np.hanning(257), axis=0)
penalty = np.transpose(hann_1d) * hann_1d
penalty = penalty / np.sum(penalty)

class Siamese(object):
	def __init__(self, batch_size):
		self.batch_size =batch_size
		self.batched_pos_x_ph = tf.placeholder(tf.float64, shape = [self.batch_size])
		self.batched_pos_y_ph = tf.placeholder(tf.float64, shape = [self.batch_size])
		self.batched_z_sz_ph = tf.placeholder(tf.float64, shape = [self.batch_size])
		self.batched_x_sz0_ph = tf.placeholder(tf.float64, shape = [self.batch_size])
		self.batched_x_sz1_ph = tf.placeholder(tf.float64, shape = [self.batch_size])
		self.batched_x_sz2_ph = tf.placeholder(tf.float64, shape = [self.batch_size])

		self.batched_x_pos_x = tf.placeholder(tf.float64, shape = [self.batch_size])
		self.batched_x_pos_y = tf.placeholder(tf.float64, shape = [self.batch_size])
		self.batched_x_target_w = tf.placeholder(tf.float64, shape = [self.batch_size])
		self.batched_x_target_h = tf.placeholder(tf.float64, shape = [self.batch_size])

		self.label = tf.placeholder(tf.float32, [self.batch_size, None, None])




	def build_tracking_graph_train(self, final_score_sz, design, env, hp, frame_sz):
		# Make a queue of file names
		# filename_queue = tf.train.string_input_producer(frame_name_list, shuffle=False, capacity=num_frames)
		# image_reader = tf.WholeFileReader()
		# # Read a whole file from the queue
		# image_name, image_file = image_reader.read(filename_queue)	    
		image = tf.placeholder(tf.float32, [self.batch_size] + frame_sz, name = "input_image") 
		# used to pad the crops
		if design.pad_with_image_mean:
			avg_chan = tf.reduce_mean(image, axis=(1, 2), name='avg_chan') ####need to change to the mean value of each img##########
		else:
			avg_chan = None
		# pad with if necessary
		single_crops_z = []
		single_crops_x = []
		for batch in range(self.batch_size):
			single_z = image[batch]

			single_pos_x_ph = self.batched_pos_x_ph[batch]
			single_pos_y_ph = self.batched_pos_y_ph[batch]
			single_z_sz_ph = self.batched_z_sz_ph[batch]

			single_x_sz0_ph = self.batched_x_sz0_ph[batch]
			single_x_sz1_ph = self.batched_x_sz1_ph[batch]
			single_x_sz2_ph = self.batched_x_sz2_ph[batch]

			frame_padded_z, npad_z = pad_frame(single_z, frame_sz, single_pos_x_ph, single_pos_y_ph, single_z_sz_ph, avg_chan[batch])
			frame_padded_z = tf.cast(frame_padded_z, tf.float32)
			# extract tensor of z_crops
			single_crops_z.append(tf.squeeze(extract_crops_z(frame_padded_z, npad_z, single_pos_x_ph, single_pos_y_ph, single_z_sz_ph, design.exemplar_sz)))
			
			single_x = image[batch]

			frame_padded_x, npad_x = pad_frame(single_x, frame_sz, single_pos_x_ph, single_pos_y_ph, single_x_sz2_ph, avg_chan[batch])
			frame_padded_x = tf.cast(frame_padded_x, tf.float32)
			# extract tensor of x_crops (3 scales)
			single_crops_x.append(tf.squeeze(extract_crops_x(frame_padded_x, npad_x, single_pos_x_ph, single_pos_y_ph, single_x_sz0_ph, single_x_sz1_ph, single_x_sz2_ph, design.search_sz)))
		z_crops = tf.stack(single_crops_z)
		x_crops = tf.stack(single_crops_x)
		x_crops_shape = x_crops.get_shape().as_list()
		x_crops = tf.reshape(x_crops, [x_crops_shape[0] * x_crops_shape[1]] + x_crops_shape[2: ])
		print("shape of single_crops_x: ", single_crops_x[0].shape, "shape of x_crops: ", x_crops.shape)
		print("shape of single_crops_z: ", single_crops_z[0].shape, "shape of z_crops: ", z_crops.shape)
		# use crops as input of (MatConvnet imported) pre-trained fully-convolutional Siamese net
		template_z, templates_x = self._create_siamese_train(x_crops, z_crops, design)
		print("shape of template_z:", template_z.shape)
		#template_z = tf.squeeze(template_z)
		template_z_list = []
		for batch in range(self.batch_size):
			template_z_list.append(template_z[batch])
			template_z_list.append(template_z[batch])
			template_z_list.append(template_z[batch])
		templates_z = tf.stack(template_z_list)
		print("shape of templates_z:", templates_z.get_shape().as_list())
		print("shape of templates_x:", templates_x.get_shape().as_list())
		# compare templates via cross-correlation
		scores = self._match_templates_train(templates_z, templates_x)
		
		
		print("shape of small score map:", scores.get_shape().as_list())
		
		"""
		scores = scores / tf.reduce_mean(scores)
		scores = tf.maximum(scores, -10)
		scores = tf.minimum(scores, 10)
		"""
		# upsample the score maps
		
		#scores = tf.Print(scores, [scores], summarize = 300)
		
		scores_up = tf.image.resize_bilinear(scores, [final_score_sz, final_score_sz], align_corners=True)
		
		"""
		score_w = scores_up.get_shape().as_list()[1]
		score_h = scores_up.get_shape().as_list()[2]
		scores_up = tf.reshape(scores_up, [self.batch_size * 3,score_w * score_h])
		scores_up = tf.nn.softmax(scores_up) * 5 - 2.5
		scores_up = tf.reshape(scores_up, [self.batch_size * 3, score_w, score_h])
		"""
		print("shape of big score map:", scores_up.get_shape().as_list())
		score = tf.squeeze(tf.stack([scores_up[i]  for i in [0 + 3 * i for i in range(self.batch_size)]]))
		
		loss = self.cal_loss(score)
		distance_to_gt, max_pos_x, max_pos_y = self.distance(score, final_score_sz, hp)
		train_step = tf.train.AdamOptimizer(hp.lr).minimize(loss)
		summary = tf.summary.scalar('distance_to_gt', distance_to_gt)
		
		return image, z_crops, x_crops, templates_z, scores_up, loss, train_step, distance_to_gt, summary, templates_x, max_pos_x, max_pos_y

	def distance(self, score, final_score_sz, hp):
		score = score  * 0.3 + 0.7 * penalty
		print("shape of scores: ", score.get_shape().as_list())
		if (self.batch_size == 1):
			score = tf.reshape(score, [1] + score.get_shape().as_list())
		oned_scores = tf.reshape(score, [self.batch_size, score.get_shape().as_list()[1] * score.get_shape().as_list()[2]])
		print("shape of 1d_scores: ", oned_scores.shape)
		
		_, max_pos = tf.nn.top_k(oned_scores, 3)
		
		max_pos = tf.reduce_mean(max_pos, axis = 1)
		print("shape of max_pos:", max_pos.shape)
		max_pos_x = max_pos // final_score_sz
		max_pos_y = max_pos % final_score_sz
		distance_to_gt = tf.reduce_mean(tf.sqrt(tf.square(final_score_sz / 2. - tf.cast(max_pos_x, tf.float32)) + tf.square(final_score_sz / 2. - tf.cast(max_pos_y, tf.float32))))
		return distance_to_gt, max_pos_x, max_pos_y


	def cal_loss(self, score):
		#select the first row in scores as score   
		loss = tf.reduce_mean(tf.log(1 + tf.exp(-score * self.label)))
		
		return loss





	def _create_siamese_train(self, net_x, net_z, design):
		filter_h = design.filter_h
		filter_w = design.filter_w
		filter_num = design.filter_num

		# loop through the flag arrays and re-construct network, reading parameters of conv and bnorm layers
		for i in range(_num_layers):
			print('> Layer '+str(i+1))
		
			####close bn for now, since batch_size is really small 
			_filtergroup_yn = np.array([0,0,0,0,0], dtype=bool)
			# set up conv "block" with bnorm and activation 
			net_x = set_convolutional_train(net_x, filter_h[i], filter_w[i], filter_num[i], _conv_stride[i],
				                filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
				                scope='conv'+str(i+1), reuse=False)
		
			# notice reuse=True for Siamese parameters sharing
			net_z = set_convolutional_train(net_z, filter_h[i], filter_w[i], filter_num[i],_conv_stride[i],
				                filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
				                scope='conv'+str(i+1), reuse=True)    
		
			# add max pool if required
			if _pool_stride[i]>0:
				print('\t\tMAX-POOL: size '+str(_pool_sz)+ ' and stride '+str(_pool_stride[i]))
				net_x = tf.nn.max_pool(net_x, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
				net_z = tf.nn.max_pool(net_z, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))


		return net_z, net_x



	def _match_templates_train(self, net_z, net_x):
		# finalize network
		# z, x are [B, H, W, C]
		print("shape_net_z:", net_z.shape)
		net_z = tf.transpose(net_z, perm=[1,2,0,3])
		net_x = tf.transpose(net_x, perm=[1,2,0,3])
		# z, x are [H, W, B, C]
		Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
		Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))
		# assert B==Bx, ('Z and X should have same Batch size')
		# assert C==Cx, ('Z and X should have same Channels number')
		net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
		net_x = tf.reshape(net_x, (1, Hx, Wx, Bx*Cx))
		net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')
		
		print("shape of net:", net_final.get_shape().as_list())
		# final is [1, Hf, Wf, BC]
		net_final = tf.concat(tf.split(net_final, 3 * self.batch_size, axis=3), axis=0)
		print("shape of net_cat:", net_final.get_shape().as_list())
		
		# final is [B, Hf, Wf, C]
		#
		net_final = tf.reduce_mean(net_final, axis=3, keepdims = True)
		#net_final = tf.Print(net_final, [net_final], summarize = 100)
		#net_final = tf.Print(net_final, [net_final])
		

		# final is [B, Hf, Wf, 1]

		##close bn for now
		if _bnorm_adjust:

			net_final = tf.layers.batch_normalization(net_final)
		print("shape of net_final:", net_final.get_shape().as_list())
		
		return net_final
		
	def _match_templates_train_normal_conv(self, net_z, net_x):
		# finalize network
		# z, x are [B, H, W, C]
		print("shape_net_z:", net_z.shape)
		net_z = tf.transpose(net_z, perm=[1,2,0,3])
		net_x = tf.transpose(net_x, perm=[1,2,0,3])
		# z, x are [H, W, B, C]
		Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
		Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))
		# assert B==Bx, ('Z and X should have same Batch size')
		# assert C==Cx, ('Z and X should have same Channels number')
		net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
		net_x = tf.reshape(net_x, (1, Hx, Wx, Bx*Cx))
		net_final = tf.nn.conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')
		
		print("shape of net:", net_final.get_shape().as_list())
		# final is [1, Hf, Wf, BC]
		net_final = tf.concat(tf.split(net_final, 3 * self.batch_size, axis=3), axis=0)
		print("shape of net_cat:", net_final.get_shape().as_list())
		
		# final is [B, Hf, Wf, C]
		#
		#net_final = tf.reduce_mean(net_final, axis=3, keepdims = True)
		#net_final = tf.Print(net_final, [net_final], summarize = 100)
		#net_final = tf.Print(net_final, [net_final])
		

		# final is [B, Hf, Wf, 1]

		##close bn for now
		if _bnorm_adjust:

			net_final = tf.layers.batch_normalization(net_final)
		print("shape of net_final:", net_final.get_shape().as_list())
		
		return net_final

