#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 08:42:38 2018

@author: tgadfort
"""

from logger import info
from fsio import setFile

import matplotlib.pyplot as plt
import seaborn as sns
    
def plotROC(perfs, outdir, ext, pp = None):
    info("Plotting ROC Curves for {0} Classifiers".format(len(perfs)))
    modelnames = perfs.keys()
    
    plt.figure()
    current_palette = sns.color_palette()
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i,modelname in enumerate(modelnames):
        perfdata = perfs[modelname]
        auc = perfdata['AUC']
        tpr = perfdata['ROC']['tpr']
        fpr = perfdata['ROC']['fpr']
        plt.plot(fpr, tpr,
                 label='{0} ({1:0.2f})'
                 ''.format(modelname, auc),
                 color=current_palette[i], linestyle='-', linewidth=3)


    title = "Receiver Operating Characteristic"
    value = "ROC"
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('')
    plt.legend(loc="lower right")
    
    
    
    if pp is not None:
        info("Saving {0} plot to multipage pdf".format(title), ind=4)
        pp.savefig()
    else:
        plotname = setFile(outdir, ".".join([value,ext]))
        info("Saving {0} plot to {1}".format(title, plotname), ind=4)
        plt.savefig(plotname)

    plt.close()
