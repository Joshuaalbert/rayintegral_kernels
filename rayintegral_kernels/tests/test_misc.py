from .common_setup import *
import tensorflow as tf
from ..misc import forward_gradients, compute_d2K
from tensorflow.python.ops.parallel_for.gradients import jacobian

def test_forward_gradients(tf_session):
    with tf_session.graph.as_default():
        #
        x = tf.constant([1.,2.])
        y = x*x + x
        g_true = 2.*x + 1.
        g = forward_gradients(y,x)[0]
        assert np.all(tf_session.run(g) == tf_session.run(g_true))


def test_compute_d2K(tf_session):
    with tf_session.graph.as_default():
        #
        x = tf.ones((2,3))
        y = tf.reduce_sum(x*x,axis=-1)
        H = compute_d2K(y,x)
        print(tf_session.run(H).shape)


