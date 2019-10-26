# coding: utf-8
# Author : Hervé Frezza-Buet

import numpy as np
import math

### retrieve classes from the name
def ovo_classes(name):
    s = name.split(' ')
    return (int(s[0]), int(s[2]))

def plot_samples(ax, X, y, color_of_label):
    size = 200
    colors = [color_of_label(l) for l in y]
    ax.scatter(X[...,0], X[...,1], s=size, c=colors, edgecolors='black', zorder=-1)
    
    
### This plots samples with support vector information.
def plot_svc_samples(ax, descs, infos, X, y,
                     name, positive_color, negative_color,
                     alphamax_color, alpha_color, C_tol=1e-10):
    size = 200
    positive_label, negative_label = ovo_classes(name)
    info = infos[name]
    desc = descs[name]
    pos = X[y==positive_label]
    neg = X[y==negative_label]
    ax.scatter(pos[...,0], pos[...,1], s=size, c=positive_color, edgecolors='black', zorder=0)
    ax.scatter(neg[...,0], neg[...,1], s=size, c=negative_color, edgecolors='black', zorder=0)

    if info['type'] in ['C-SVC', 'nu-SVC'] :
        C_supports    = None
        notC_supports = None
        supports = desc[1]
        C = info['alphamax']
        C_supports    = np.array([x for (alpha,x) in supports if math.fabs(math.fabs(alpha)-C) <= C_tol])
        notC_supports = np.array([x for (alpha,x) in supports if math.fabs(math.fabs(alpha)-C) > C_tol])
        
        if C_supports is not None and C_supports.shape[0] > 0:
            ax.scatter(   C_supports[...,0],    C_supports[...,1], s=size, c=alphamax_color,    linewidth=3, marker='x', zorder=1)
        if notC_supports is not None and notC_supports.shape[0] > 0:
                ax.scatter(notC_supports[...,0], notC_supports[...,1], s=size, c=alpha_color, linewidth=3, marker='x', zorder=1)
                

        

def plot_svc_separation(ax, seps, infos, name, xlim, ylim, fontsize=15):
    sep  = seps[name]
    info = infos[name]
    if info['type'] in ['C-SVC', 'nu-SVC'] :
        nb_steps = 100
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], nb_steps),
                             np.linspace(ylim[0], ylim[1], nb_steps))
        Z = np.array([sep(x) for x in np.c_[xx.ravel(), yy.ravel()]])
        Z = Z.reshape(xx.shape)
        rho = info['rho']
        cs = ax.contour(xx, yy, Z,
                        colors='black',
                        levels=[-rho, 0, rho], 
                        linestyles=['dashed', 'solid', 'solid'],
                        linewidths=[1,3,1],
                        zorder=-2)
        ax.clabel(cs, cs.levels, fontsize=fontsize, fmt={-1: r'$h(x)=-{}$'.format(rho), 0:r'$h(x)=0$', 1:r'$h(x)={}$'.format(rho)})


    
def plot_svc_partition(ax, classifier, xlim, ylim, colors):
    nb_steps = 100
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], nb_steps),
                         np.linspace(ylim[0], ylim[1], nb_steps))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]) # np.c[[x,x,...], [y,y,...] = [(x,y), (x,y), ...]
    Z = Z.reshape(xx.shape)
    levels = [x for x in np.sort(classifier.classes_)-.5]
    last = levels[-1]+1
    levels.append(last)
    ax.contourf(xx, yy , Z, levels, colors=colors, zorder=-3)
    
