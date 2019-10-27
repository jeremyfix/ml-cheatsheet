#!/usr/bin/env python3

# Extern modules
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm
import sklearn.datasets
# Local modules
import svm_tools
import plot_tools


def exampleCSVC_moon(n_samples, C, noise, sigma, filename):
    X, y = sklearn.datasets.make_moons(n_samples=n_samples,
                                       noise=noise)
    gamma = 1./(2.0 * sigma**2)
    clf = sklearn.svm.SVC(C=C,
                          kernel='rbf', gamma=gamma,
                          tol=1e-5,
                          decision_function_shape='ovo')
    clf.fit(X, y)
    sep, desc, info = svm_tools.svc_ovo_separators(clf,
                                                   svm_tools.gaussian_kernel(gamma),
                                                   X)
    fig = plt.figure(figsize=(12, 10))

    xlim = [-1.5, 2.5]
    ylim = [-1.5, 1.5]

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    plot_tools.plot_svc_samples(ax, desc, info, X, y,
                                '0 vs 1', 'white', 'gray',
                                'black', 'red')
    plot_tools.plot_svc_separation(ax, sep, info, '0 vs 1', xlim, ylim)
    plot_tools.plot_svc_partition(ax, clf,
                                  xlim, ylim,
                                  [(.8, .8, 1.), (.8, 1., .8)])
    plt.savefig(filename, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    exampleCSVC_moon(100, 1, 0.1, 0.4, "csvc001.png")
    exampleCSVC_moon(100, 1000, 0.25, 0.4, "csvc002.png")
    exampleCSVC_moon(400, 0.1, 0.25, 0.4, "csvc003.png")
