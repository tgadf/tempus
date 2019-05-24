#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 17:49:12 2018

@author: tgadfort
"""

from logger import info

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score    
        
def getPerformance(y_truth, testResults):
    info("Getting classifier performance", ind=2)
        
    y_prob = testResults['prob']
    y_pred = testResults['label']

    retval = {}
    
    precision, recall, pr_thresholds = precision_recall_curve(y_truth, y_prob)
    retval["PR"] = {"precision": precision, "recall": recall, "thresholds": pr_thresholds}
    info("Got precision, recall", ind=6)
    
    info("Getting fpr, tpr, roc", ind=6)
    fpr, tpr, roc_thresholds = roc_curve(y_truth, y_prob)
    retval["ROC"] = {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds}
    info("Got ROC", ind=6)
    
    cfm = confusion_matrix(y_truth, y_pred)
    tn, fp, fn, tp = cfm.ravel()
    retval["Confusion"] = {"matrix": cfm, "tn": tn, "tp": tp, "fn": fn, "fp": fp}
    info("Got confusion matrix", ind=6)
    
    mtw = matthews_corrcoef(y_truth, y_pred)
    retval["Matthews"] = mtw
    info("Matthews Coefficient: {0}".format(round(mtw, 3)), ind=6)
    
    kpa = cohen_kappa_score(y_truth, y_pred)
    retval["Kappa"] = kpa
    info("Kappa: {0}".format(round(kpa, 3)), ind=6)
    
    acc = accuracy_score(y_truth, y_pred)
    retval["Accuracy"] = acc
    info("Accuracy: {0}".format(round(acc, 3)), ind=6)
    
    f1s = f1_score(y_truth, y_pred)
    retval["F1"] = f1s
    info("F1: {0}".format(round(f1s, 3)), ind=6)
    
    hml = hamming_loss(y_truth, y_pred)
    retval["HammingLoss"] = hml
    info("Hamming Loss: {0}".format(round(hml, 3)), ind=6)
    
    jss = jaccard_similarity_score(y_truth, y_pred)
    retval["JaccardScore"] = jss
    info("Jaccard Similarity: {0}".format(round(jss, 3)), ind=6)
    
    lls = log_loss(y_truth, y_prob)
    retval["LogLoss"] = lls
    info("Log Loss: {0}".format(round(lls, 3)), ind=6)
    
    pcs = precision_score(y_truth, y_pred)
    retval["Precision"] = pcs
    info("Precision: {0}".format(round(pcs, 3)), ind=6)
    
    rcs = recall_score(y_truth, y_pred)
    retval["Recall"] = rcs
    info("Recall: {0}".format(round(rcs, 3)), ind=6)
    
    zol = zero_one_loss(y_truth, y_pred)
    retval["ZeroOneLoss"] = zol
    info("Zero One Loss: {0}".format(round(zol, 3)), ind=6)
    
    bsl = brier_score_loss(y_truth, y_prob)
    retval["BrierLoss"] = bsl
    info("Brier Loss: {0}".format(round(bsl, 3)), ind=6)
    
    aps = average_precision_score(y_truth, y_prob)
    retval["AveragePrecision"] = aps
    info("Average Precision: {0}".format(round(aps, 3)), ind=6)
    
    auc = roc_auc_score(y_truth, y_prob)
    retval["AUC"] = auc
    info("AUC: {0}".format(round(auc,3)), ind=6)
    
    return retval
