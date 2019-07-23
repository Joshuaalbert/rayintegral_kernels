import numpy as np
import tensorflow as tf

from rayintegral_kernels import float_type
from rayintegral_kernels.kernel import RandomKernel, M52, TrapezoidKernel
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


def test_rbf_taylor():
    with tf.Session(graph=tf.Graph()) as sess:
        N = 2
        x = tf.constant([[0., 0., t] for t in np.linspace(0., 60., N)], float_type)
        k = tf.constant([[0., np.sin(theta),np.cos(theta)] for theta in np.linspace(-4.*np.pi/180., 4.*np.pi/180., N)], float_type)
        k /= tf.linalg.norm(k, axis=1, keepdims=True)
        #
        X = tf.concat([k,x],axis=1)
        theta = tf.constant([1., 10.],float_type)
        a = tf.constant(200., float_type)
        b = tf.constant(100., float_type)
        mu = None
        ref_location = X[0,3:6]
        ref_direction = X[0,0:3]

        ref_kern = RandomKernel(M52(theta), 2000, a, b, mu=mu, ref_location=ref_location, ref_direction=ref_direction,
                                obs_type='DTEC', ionosphere_type='flat')
        ref_K = ref_kern.K(X, X)

        ref_g = tf.gradients(ref_K,[theta])[0]
        ref_K = sess.run(ref_K)
        ref_g = sess.run(ref_g)
        F = []
        R = [4,5,6,7,8,9]
        import pylab as plt
        plt.imshow(ref_K)
        plt.colorbar()
        plt.show()
        for res in R:
            test_kern = TrapezoidKernel(M52(theta), res, a, b, mu=mu, ref_location=ref_location,
                                     ref_direction=ref_direction,
                                     obs_type='DTEC', ionosphere_type='flat')
            from timeit import default_timer
            K = test_kern.K(X, X)
            g = tf.gradients(K,[theta])[0]
            t0 = default_timer()
            K = sess.run(K)
            print(K)
            print((default_timer()-t0))
            t0 = default_timer()
            g = sess.run(g)
            print((default_timer() - t0))
            plt.imshow(K)
            plt.colorbar()
            plt.show()
            print(ref_g,g, ref_g-g)

            f = np.mean(np.abs(ref_K - K))
            F.append(f)

        plt.plot(R, F)

        plt.show()

        # ref_kern = RandomKernel(RBF(theta), 2000, a, b, mu=mu, ref_location = ref_location, ref_direction = ref_direction, obs_type='DTEC', ionosphere_type='flat')
        # g = tf.gradients(tf.reduce_sum(ref_kern.K(X,X)), [theta])[0]
        # ref_g = sess.run(g)
        # F = []
        # R = [2,3,4,5]
        # for res in R:
        #     test_kern = TaylorKernel(RBF(theta), res, a, b, mu=mu, ref_location=ref_location, ref_direction=ref_direction,
        #                  obs_type='DTEC', ionosphere_type='flat')
        #     g = tf.gradients(tf.reduce_sum(test_kern.K(X, X)), [theta])[0]
        #
        #     f = np.mean(np.abs(ref_g - sess.run(g)))
        #     F.append(f)
        # import pylab as plt
        # plt.plot(R,F)
        #
        # plt.show()