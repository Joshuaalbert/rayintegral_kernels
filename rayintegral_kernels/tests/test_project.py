import tensorflow as tf
from .common_setup import *

def test_tensorflow(tf_session):
    with tf_session.graph.as_default():
        assert tf_session.run(tf.constant(0)) == 0