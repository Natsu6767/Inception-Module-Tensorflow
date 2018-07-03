import tensorflow as tf

def fc_layer(x, input_size, output_size, name, relu = True):

	"""Create a fully connected layer."""
	
	with tf.variable_scope(name) as scope:

		# Create tf variables for the weights and biases.
		W = tf.get_variable('weights', shape = [input_size, output_size],
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

		b = tf.get_variable('biases', shape = [output_size], initializer = tf.constant_initializer(0.0))

		# Matrix multiply weights and inputs and add biases.
		z = tf.nn.bias_add(tf.matmul(x, W), b)

		if relu:
			# Apply ReLu non linearity.
			a = tf.nn.relu(z)
			return a

		else:
			return z