import tensorflow as tf

'''
additional custom metrics for monitoring training process
'''

def tp(y_true, y_pred):
	y_true = tf.keras.backend.argmax(y_true)
	y_true = tf.cast(y_true, tf.float64)
	y_pred = tf.keras.backend.argmax(y_pred)
	y_pred = tf.cast(y_pred, tf.float64)
	
	tp = tf.keras.backend.sum(y_true * y_pred)
	return tp

def fp(y_true, y_pred):
	y_true = tf.keras.backend.argmax(y_true)
	y_true = tf.cast(y_true, tf.float64)
	y_pred = tf.keras.backend.argmax(y_pred)
	y_pred = tf.cast(y_pred, tf.float64)

	neg_y_true = 1 - y_true
	fp = tf.keras.backend.sum(neg_y_true * y_pred)
	return fp

def tn(y_true, y_pred):
	y_true = tf.keras.backend.argmax(y_true)
	y_true = tf.cast(y_true, tf.float64)
	y_pred = tf.keras.backend.argmax(y_pred)
	y_pred = tf.cast(y_pred, tf.float64)

	neg_y_true = 1 - y_true
	neg_y_pred = 1 - y_pred
	tn = tf.keras.backend.sum(neg_y_true * neg_y_pred)
	return tn
    
def fn(y_true, y_pred):
	y_true = tf.keras.backend.argmax(y_true)
	y_true = tf.cast(y_true, tf.float64)
	y_pred = tf.keras.backend.argmax(y_pred)
	y_pred = tf.cast(y_pred, tf.float64)

	neg_y_pred = 1 - y_pred
	fn = tf.keras.backend.sum(y_true * neg_y_pred)
	return fn