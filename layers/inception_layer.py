import tensorflow as tf

from conv_layer import conv_layer
from max_pool import max_pool

def inception_layer(x, conv_1_size, conv_3_reduce_size,
					conv_3_size, conv_5_reduce_size,
					conv_5_size, pool_proj_size,
					name = 'inception'):

	""" Create an Inception Layer """

	with tf.variable_scope(name) as scope:

		conv_1 = conv_layer(x, filter_height = 1, filter_width = 1,
							num_filters = conv_1_size, name = '{}_1x1'.format(name))

		conv_3_reduce = conv_layer(x, filter_height = 1, filter_width = 1,
							num_filters = conv_3_reduce_size, name = '{}_3x3_reduce'.format(name))

		conv_3 = conv_layer(conv_3_reduce, filter_height = 3, filter_width = 3,
							num_filters = conv_3_size, name = '{}_3x3'.format(name))

		conv_5_reduce = conv_layer(x, filter_height = 1, filter_width = 1,
							num_filters = conv_5_reduce_size, name = '{}_5x5_reduce'.format(name))

		conv_5 = conv_layer(conv_5_reduce, filter_height = 5, filter_width = 5,
							num_filters = conv_5_size, name = '{}_5x5'.format(name))

		pool = max_pool(x, stride =1, padding = 'SAME', name = '{}_pool'.format(name))

		pool_proj = conv_layer(pool, filter_height = 1, filter_width = 1,
							num_filters = pool_proj, name = '{}_pool_proj'.format(name))

		return tf.concat([conv_1, conv_3, conv_55, pool_proj], axis = 3, name='{}_concat'.format(name))
