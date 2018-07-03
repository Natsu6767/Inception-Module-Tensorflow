import tensorflow as tf

def max_pool(x, name, filter_height = 3, filter_width = 3,
	stride = 2, padding = 'VALID'):

	"""Create a max pooling layer."""

	return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
		strides = [1, stride, stride, 1], padding = padding, name = name)