from rayintegral_kernels.kernel import TrapezoidKernel, RandomKernel, RBF, M52, M32, M12, RQ
import tensorflow as tf
from rayintegral_kernels import float_type
import numpy as np
import pylab as plt

def main():
    """
    D(R) = <(phi(r) - phi(r+R))^2>
         = <K^2(TEC(r) - TEC(r + R))^2>
         = K^2<DTEC(r, -R)^2>

    :return:
    """


    fig, ax = plt.subplots(1,1,figsize=(6,6))
    baselines = 10 ** np.linspace(-1., 2, 100)



    k_obs = []
    for _ in range(500):
        ds = np.random.uniform(5., 30.)
        beta = np.random.normal(1.89, 0.1)
        k_obs.append((baselines / ds) ** (beta))
    ax.plot(baselines, np.mean(k_obs, axis=0), lw=2., color='black', label='Mevius+ 2016')
    ax.fill_between(baselines, np.percentile(k_obs, 5, axis=0), np.percentile(k_obs, 95, axis=0), color='yellow',
                     alpha=0.5)

    k_turb = (baselines / 10.) ** (5. / 3.)
    ax.plot(baselines, k_turb, color='green', lw=2., label=r'Kolmogorov $5/3$')

    with tf.Session() as sess:

        x = tf.placeholder(float_type, shape=(3,))
        k = tf.placeholder(float_type, shape=(3,))
        khat = k/tf.linalg.norm(k)
        X = tf.concat([khat[None,:],x[None,:]],axis=1)#1,6
        theta = tf.constant([6., 14., 1./3.], float_type)
        int_kern = M32(theta)# + RBF(theta/200.)
        kern = TrapezoidKernel(int_kern,
                               20,
                               tf.constant(250., float_type),
                               tf.constant(100., float_type),
                               obs_type='DTEC')
        K = kern.K(X,X)

        xy = []
        z = []

        for b in baselines:
            xy.append([np.concatenate([[b, 0.,0.]], axis=0),
                       [0.,0.,1.]])
                       # np.concatenate([0.0 * np.random.normal(size=2), [1.]],axis=0)
            xy[-1][1] /= np.linalg.norm(xy[-1][1])

            np_K = sess.run(K,{x:xy[-1][0],
                               k:xy[-1][1]})
            z.append(np.sqrt(np_K[0,0]))
        xy = np.array(xy)
        z = 8.448e6*np.array(z)/150e6

        ax.plot(xy[:,0,0],z**2, ls='dotted', lw=2., color='blue', label='dawn')

        with tf.Session() as sess:

            x = tf.placeholder(float_type, shape=(3,))
            k = tf.placeholder(float_type, shape=(3,))
            khat = k / tf.linalg.norm(k)
            X = tf.concat([khat[None, :], x[None, :]], axis=1)  # 1,6
            theta = tf.constant([3., 17., 1. / 3.], float_type)
            int_kern = RBF(theta)  # + RBF(theta/200.)
            kern = TrapezoidKernel(int_kern,
                                   20,
                                   tf.constant(350., float_type),
                                   tf.constant(200., float_type),
                                   obs_type='DTEC')
            K = kern.K(X, X)

            xy = []
            z = []

            for b in baselines:
                xy.append([np.concatenate([[b, 0., 0.]], axis=0),
                           [0., 0., 1.]])
                # np.concatenate([0.0 * np.random.normal(size=2), [1.]],axis=0)
                xy[-1][1] /= np.linalg.norm(xy[-1][1])

                np_K = sess.run(K, {x: xy[-1][0],
                                    k: xy[-1][1]})
                z.append(np.sqrt(np_K[0, 0]))
            xy = np.array(xy)
            z = 8.448e6 * np.array(z) / 150e6

            ax.plot(xy[:, 0, 0], z ** 2, ls='dashed', lw=2., color='pink', label='dusk')
            print(z)



    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(0.1,100.)
    ax.set_ylim(1e-4,1e2)
    ax.set_ylabel(r"$\mathrm{Var}[\phi_{\rm 150}]$ [rad$^2$]")
    ax.set_xlabel("Baseline [km]")
    # plt.tricontour(xy[:,0,0],xy[:,1,0], z,levels=10)
    # plt.title("Over x")
    ax.legend()
    plt.savefig("/home/albert/Documents/structure_function.pdf")
    plt.show()
    plt.subplot(projection='3d')
    plt.scatter(xy[:, 1, 0], xy[:, 1, 1], c=z, alpha=0.5, marker='+')
    plt.tricontour(xy[:, 1, 0], xy[:, 1, 1], z, levels=10)
    plt.title("Over k")
    plt.show()

    diff_scale = []
    thetas = np.linspace(0., np.pi/6., 100)
    with tf.Session() as sess:

        x = tf.placeholder(float_type, shape=(3,))
        k = tf.placeholder(float_type, shape=(3,))
        khat = k/tf.linalg.norm(k)
        X = tf.concat([khat[None,:],x[None,:]],axis=1)#1,6
        theta = tf.constant([10., 14., 1./3.], float_type)
        int_kern = M32(theta)# + RBF(theta/200.)
        kern = TrapezoidKernel(int_kern,
                               20,
                               tf.constant(250., float_type),
                               tf.constant(100., float_type),
                               obs_type='DTEC')
        K = kern.K(X,X)

        for theta in thetas:

            xy = []
            z = []

            for b in baselines:
                xy.append([np.concatenate([[b, 0.,0.]], axis=0),
                           [0.,np.sin(theta),np.cos(theta)]])
                           # np.concatenate([0.0 * np.random.normal(size=2), [1.]],axis=0)
                xy[-1][1] /= np.linalg.norm(xy[-1][1])

                np_K = sess.run(K,{x:xy[-1][0],
                                   k:xy[-1][1]})
                z.append(np.sqrt(np_K[0,0]))
            xy = np.array(xy)
            z = 8.448e6*np.array(z)/150e6
            ds = np.interp(1., z, xy[:,0,0])
            diff_scale.append(ds)
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.plot(np.array(thetas)*180/np.pi, diff_scale, lw=2., color='blue')
        ax.grid()
        ax.set_ylabel(r'$r_{\rm scale}$ [km]')
        ax.set_xlabel('Zenith angle [degrees]')
        plt.savefig('/home/albert/Documents/diffractive_scale.pdf')
        plt.show()



if __name__ == '__main__':
    main()