import tensorflow as tf

def MFE_step_static(last_MFE_values, adjacency_mat):

	"""
	Input:
	last_mfe_values - Tensor (batch_size, n_points, latent_dim)
	adjacency_mat - Tensor (batch_size, n_points, n_points)
	"""
	rtn_MFE_values=[]
	for i in range(n_points):
		cur_adj_mat=tf.expand_dims(adjacency_mat[:,i,:],axis=-1)
		cur_MFE_mult=last_MFE_values*cur_adj_mat
		# new_MFE_point_val=tf.reduce_sum(cur_MFE_mult,axis=1)
		new_MFE_point_val=tf.reduce_mean(cur_MFE_mult,axis=1,keep_dims=True)
		rtn_MFE_values.append(new_MFE_point_val)
	return tf.concat(rtn_MFE_values,axis=1)

def vectorized_MFE_step_static(last_MFE_values, adjacency_mat):
	"""
	Input:
	last_mfe_values - Tensor (batch_size, n_points, latent_dim)
	adjacency_mat - Tensor (batch_size, n_points, n_points)
	"""
	expanded_point_value = tf.expand_dims(point_values,axis=1)
	expanded_adj_mat = tf.expand_dims(adjacency_mat,axis=-1)
	MFE_mult=expanded_point_value*expanded_adj_mat
	MFE=tf.reduce_mean(MFE_mult,axis=2)
	# MFE=tf.reduce_sum(MFE_mult,axis=2)
	return MFE

def zero_state(shape):
	"""
	Input:
	shape - tuple of integers (batch_size, n_points, latent_dim)
	Output:
	state - zero Tensor (batch_size, n_points, latent_dim)
	"""
	return tf.zeros(shape)

def MFE_static(point_values,adjacency_matrix,last_MFE_values):

	mfe=MFE_step_static(last_MFE_values,adjacency_matrix)
	emb_dim=mfe.shape[-1]
	with tf.variable_scope('mfe_embedding'):
		Wx=tf.get_variable('Wx',shape=[point_values.shape[-1],emb_dim],initializer=tf.contrib.layers.xavier_initialization())
		bx=tf.get_variable('bx',initializer=tf.zeros([emb_dim]))
		WL=tf.get_variable('WL',shape=[emb_dim,emb_dim],initializer=tf.contrib.layers.xavier_initialization())
		bL=tf.get_variable('bL',initializer=tf.zeros([emb_dim]))
	pv=tf.reshape(point_values,[-1,point_values.shape[-1]])
	mfe=tf.reshape(mfe,[-1,emb_dim])
	preshape=tf.matmul(pv,Wx)+bx+tf.matmul(mfe,WL)+bL
	return tf.nn.relu(tf.reshape(preshape,[-1,adjacency_matrix.shape[-1],emb_dim]))
