from .common_setup import *

from ..kernel import RandomKernel, TrapezoidKernel, RBF, RQ

def test_RBF(tf_session):
    with tf_session.graph.as_default():
        x = 20. * tf.random.normal([4, 3], dtype=float_type)
        k = tf.random.normal([4, 3], dtype=float_type)
        k /= tf.linalg.norm(k, axis=-1, keepdims=True)
        X = tf.concat([k, x], axis=1)

        theta = tf.constant([1., 20.], float_type)
        rbf = RBF(theta)
        kern_trap = TrapezoidKernel(rbf, 15, tf.constant(200., float_type),
                               tf.constant(100., float_type))
        kern_rand = RandomKernel(rbf, 20000, tf.constant(200., float_type),
                                    tf.constant(100., float_type))

        print(tf_session.run([kern_trap.K(X,X),
                              kern_rand.K(X,X)]))

def test_RQ(tf_session):
    import pylab as plt
    with tf_session.graph.as_default():
        theta_pl = tf.placeholder(float_type, shape=(3,))
        rq = RQ(theta_pl)
        res = rq.apply(tf.cast(tf.linspace(0., 10., 100)[:,None], float_type))
        p_array = [1./3.]#1./np.linspace(0.01, 100., 10)
        for p in p_array:
            K = tf_session.run(res, {theta_pl:[1., 1., p]})
            print(K)
            plt.plot(np.log(K),label=p)

        plt.legend()
        plt.show()
