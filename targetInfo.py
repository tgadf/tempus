#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 09:18:19 2017

@author: tgadfort
"""
#import tqdm

import colInfo
from logger import info

def getTargetType(targetData):    
    info('Checking target type', ind=3)
    ctype = colInfo.getColType(targetData)
    return ctype


def isRegression(pType):
    if pType == 'regress' or pType == 'regression':
        return True
    return False
    
def isClassification(pType):
    if pType == 'classify' or pType == 'classification':
        return True
    return False
    
def isClustering(pType):
    if pType == 'cluster':
        return True
    return False
    

def getProblemType(targetData):
    info('Checking problem type', ind=2)
    
    ttype = getTargetType(targetData)
    if ttype == str:
        raise ValueError("String type is not allowed for target data.")
    if ttype == bool:
        raise ValueError("Bool type is not allowed for target data.")

    sampleData = targetData.sample(frac=0.1)
    nUnique    = len(sampleData.unique())
    if nUnique >= 2:
        return 'regress'
    else:
        return 'classify'
   
    
def getTargetNames(config):
    if isClassification(config['problem']):
        targetcol = config['target']['colname']
        nontarget = "Non"+targetcol
        return [targetcol, nontarget]
    else:
        return [None, None]
