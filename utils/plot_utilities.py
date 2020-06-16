import matplotlib.pyplot as plt
import numpy as np


def subplot_safemdp(n, m, myfunc, source):
    """ Define a function to create a subplot"""
    plt.subplot(321)
    plt.contourf(myfunc[:,1].reshape(n,m))
    plt.plot(int(source%m), int(source/m), 'ro')
    plt.colorbar()
    plt.subplot(322)
    plt.contourf(myfunc[:,2].reshape(n,m))
    plt.plot(int(source%m), int(source/m), 'ro')
    plt.colorbar()
    plt.subplot(323)
    plt.contourf(myfunc[:,3].reshape(n,m))
    plt.plot(int(source%m), int(source/m), 'ro')
    plt.colorbar()
    plt.subplot(324)
    plt.contourf(myfunc[:,4].reshape(n,m))
    plt.plot(int(source%m), int(source/m), 'ro')
    plt.colorbar()
    plt.subplot(325)
    plt.contourf(myfunc[:,0].reshape(n,m))
    plt.plot(int(source%m), int(source/m), 'ro')
    plt.colorbar()
    plt.show()


def contour_maps(x, p_safe, R_eb, source, opt_path):
    """ Create a contour map for safety function or reward function """
    n, m = x.world_shape

    np_opt_path = np.zeros((len(opt_path), 2))
    for i in range(len(opt_path)):
        np_opt_path[i,0] = int(opt_path[i]%m)
        np_opt_path[i,1] = int(opt_path[i]/m)

    plt.contourf(p_safe[:,1].reshape(n,m))
    plt.plot(int(source%m), int(source/m), 'ro')
    plt.plot(np_opt_path[:,0], np_opt_path[:,1], 'm-')
    plt.colorbar()
    plt.show()

    subplot_safemdp(n, m, p_safe, source)
    #subplot_safemdp(n, m, R_eb[:-1,:], source)


def performance_metrics(path, x, true_S_hat_epsilon, true_S_hat, h_hard):
    """
    Parameters
    ----------
    path: np.array
        Nodes of the shortest safe path
    x: SafeMDP
         Instance of the SafeMDP class for the mars exploration problem
     true_S_hat_epsilon: np.array
        True S_hat if safety feature is known up to epsilon and h is used
    true_S_hat: np.array
        True S_hat if safety feature is known with no error and h_hard is used
    h_hard: float
        True safety thrshold. It can be different from the safety threshold
        used for classification in case the agent needs to use extra caution
        (in our experiments h=25 deg, h_har=30 deg)

    Returns
    -------
    unsafe_transitions: int
        Number of unsafe transitions along the path
    coverage: float
        Percentage of coverage of true_S_hat_epsilon
    false_safe: int
        Number of misclassifications (classifing something as safe when it
        acutally is unsafe according to h_hard )
    """

    # Count unsafe transitions along the path
    path_altitudes = x.altitudes[path]
    unsafe_transitions = np.sum(-np.diff(path_altitudes) < h_hard)

    # Coverage
    max_size = float(np.count_nonzero(true_S_hat_epsilon))
    coverage = 100 * np.count_nonzero(np.logical_and(x.S_hat,
                                                true_S_hat_epsilon))/max_size
    # False safe
    false_safe = np.count_nonzero(np.logical_and(x.S_hat, ~true_S_hat))

    return unsafe_transitions, coverage, false_safe
