#!/usr/bin/env python3

# Extern modules
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm
import sklearn.datasets
# Local modules
import svm_tools
import plot_tools


def train_classifier(build_clf,
                     n_samples):
    X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.1)

    # Let us learn a linear C-SVC
    classifier = build_clf()
    classifier.fit(X, y)

    return classifier, (X, y)


def example001():
    build_clf = lambda: sklearn.svm.SVC(C=1,
                                        kernel='rbf', gamma=1/(2.0 * 0.4**2),
                                        tol=1e-5,
                                        decision_function_shape='ovo')
    clf, (X, y) = train_classifier(build_clf, 100)
    sep, desc, info = svm_tools.svc_ovo_separators(clf,
                                                   svm_tools.gaussian_kernel(clf.gamma), X)
    fig = plt.figure(figsize=(12, 10))

    xlim = [-1.5, 2.5]
    ylim = [-1.5, 1.5]

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    plot_tools.plot_svc_samples(ax, desc, info, X[:100], y[:100],
                                '0 vs 1', 'white', 'gray',
                                'black', 'red')
    plot_tools.plot_svc_separation(ax, sep, info, '0 vs 1', xlim, ylim)
    plot_tools.plot_svc_partition(ax, clf,
                                  xlim, ylim,
                                  [(.8, .8, 1.), (.8, 1., .8)])
    plt.savefig('csvc001.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    example001()
