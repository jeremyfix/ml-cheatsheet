# coding: utf-8
# Author : Hervé Frezza-Buet

import math
import numpy as np

import sklearn.svm

def empirical_risk(classifier, X, y):
    ypred = classifier.predict(X)
    return sum(y != ypred)/float(X.shape[0])

def linear_kernel(x, z):
    return np.dot(x, z)

def gamma_of_sigma(sigma) :
    return 1/(2.0*sigma**2)

def gaussian_kernel(gamma):
    def k(x, z) :
        diff = x - z
        return math.exp(-np.dot(diff,diff)*gamma)
    return k

###
###
### Handling the supports is quite intricated with libsvm,
### the following methods enables retreiving all the svm
### computation.
###
###

### For internal use.
def ovo_name(class1, class2):
    return '{} vs {}'.format(class1,class2)

### For internal use.
### this generates all the "i vs j" names.
def ovo_names(nb_classes):
    classes = range(nb_classes)
    for (idx,c1) in enumerate(classes):
        for c2 in classes[idx+1:] :
            yield ovo_name(c1,c2)

### For internal use.
### this generates all the "i vs j" names where the_class is one of the i or j.
def ovo_other_names(the_class, nb_classes):
    for c in range(nb_classes) :
        if c < the_class:
            yield ovo_name(c, the_class)
        elif c > the_class:
            yield ovo_name(the_class, c)

### This returns two dictionnaries whose keys are 'i vs j', i < j, and a third one.
### The first one contains the separators (i.e. the functions).
### The second one contains the separator descriptions (b, [(alpha_1,x1), (alpha_2, x_2), ...]
### The third dictionary contains display informations.
def svc_ovo_separators(classifier, kernel, X):
    nb_classes  = len(classifier.classes_)
    desc        = dict()                                                   # This is the resulting separator descriptions
    offsets     = (offset for offset in classifier.intercept_)
    for name in ovo_names(nb_classes) :                                    # we init the dictionary content as empty lists.
        desc[name] = (next(offsets),[])                                    # desc['i vs j'] = (b, [(alpha_1,x1), (alpha_2, x_2), ...])
    nb_supports = (n for n in classifier.n_support_)                       # nb_sup_class1, nb_sup_class2, ...
    support_idx = (idx for idx in classifier.support_)                     # 1,4,18,2,.... the ranks of the supports in the input set
    alphas_s    = (coefs for coefs in classifier.dual_coef_.transpose())   # read the docs, it is intricated.
    for the_class in range(nb_classes) :                                   # for each class
        for sup_rank in range(next(nb_supports)) :                         # for each support for that class
            sup_idx = next(support_idx)                                    # The idx of the support vector concerned by the coefs
            coefs   = (coef for coef in next(alphas_s))                    # the coefs for each classifier for that support.
            for name in ovo_other_names(the_class, nb_classes) :
                coef = next(coefs)
                if coef != 0 :
                    desc[name][1].append((coef, X[sup_idx]))
    # Let us build up a dictionary of functions.
    sep = dict()

    # The default parameters in the lambda function is a tricky way to perform a correct capture...
    for key, val in desc.items():
        sep[key] = lambda x, b=val[0], alphaxi=val[1] : b + np.array([alpha*kernel(xi,x) for alpha, xi in alphaxi]).sum()

    infos = dict()
    for key,val in desc.items() : 
        info = dict()
        if isinstance(classifier, sklearn.svm.SVC) :
            info['type'] = 'C-SVC'
            info['alphamax'] = classifier.C
        elif isinstance(classifier, sklearn.svm.NuSVC) :
            info['type'] = 'nu-SVC'
            info['alphamax'] = max([alpha for alpha, xi in val[1]])
        else :
            info['type'] = None
        info['rho']  = 1
        infos[key] = info
        
        
    return sep, desc, infos
