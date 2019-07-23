import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

def forward_gradients(ys, xs, grad_xs=None, gate_gradients=False):
    """Forward-mode pushforward analogous to the pullback defined by tf.gradients.
    With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
    the vector being pushed forward.
    
    Taken from https://github.com/renmengye/tensorflow-forward-ad
    """
    if type(ys) == list:
        v = [tf.ones_like(yy) for yy in ys]
    else:
        v = tf.ones_like(ys)  # dummy variable
    g = tf.gradients(ys, xs, grad_ys=v)
    return tf.gradients(g, v, grad_ys=grad_xs)

def flatten_batch_dims(t, num_batch_dims=None):
    """
    Flattens the first `num_batch_dims`
    :param t: Tensor [b0,...bB, n0,...nN]
        Flattening happens for first `B` dimensions
    :param num_batch_dims: int, or tf.int32
        Number of dims in batch to flatten. If None then all but last. If < 0 then count from end.
    :return: Tensor [b0*...*bB, n0,...,nN]
    """
    shape = tf.shape(t)
    if num_batch_dims is None:
        num_batch_dims =  - 1
    out_shape = tf.concat([[-1], shape[num_batch_dims:]],axis=0)
    return tf.reshape(t,out_shape)

def compute_d2K(K, lamda):
    with tf.name_scope('compute_d2K',values=[K,lamda]):
        shape = tf.shape(K)
        K_flatbatch = flatten_batch_dims(K,-1)
        J = batch_jacobian(K_flatbatch,lamda)
        H = batch_jacobian(J,lamda)
        H = tf.reshape(H,tf.concat([shape] + 2*[tf.shape(lamda)[-1:]],axis=0))
        return H