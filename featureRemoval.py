#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 18:22:30 2018

@author: tgadfort
"""

from logger import info
from pandas import concat
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import f_regression, mutual_info_regression

from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

from sklearn.covariance import EmpiricalCovariance, MinCovDet

from targetinfo import isRegression, isClassification
from colInfo import getNcols

def removeLowVarianceData(pddf, threshold):
    info("Removing data with low variance", ind=4)
    
    # Binary data
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit_transform(pddf)
    
    
    
def removeLowImportanceData(Xdata, config, estimator = None, y = None):
    info("Removing features that were not important.", ind=4)
    
    selectionConfig = config['features']['Selection']
    importanceThreshold  = selectionConfig['importance']

    if estimator is None:
        if y is None:
            raise ValueError("Need target data to determine feature importance.")
            
        problemType = config['problem']
        if isClassification(problemType):
            estimator = ExtraTreesClassifier()
            estimator = estimator.fit(Xdata, y)
        if isRegression(problemType):
            estimator = ExtraTreesRegressor()
            estimator = estimator.fit(Xdata, y)

    sfm = SelectFromModel(estimator, threshold=importanceThreshold)
    
    info("Feature data has "+getNcols(Xdata, asStr=True)+" columns.", ind=6)
    sfm.fit_transform(Xdata)
    info("Feature data now has "+getNcols(Xdata, asStr=True)+" columns.", ind=6)
    
    
    
def removeLowQualityData(pddf, config):
    info("Removing features that are not correlated with target.", ind=4)
    
    # SelectKBest removes all but the k highest scoring features
    # SelectPercentile removes all but a user-specified highest scoring percentage of features
    #   using common univariate statistical tests for each feature: false positive rate 
    # SelectFpr, false discovery rate SelectFdr, or family wise error SelectFwe.
    # GenericUnivariateSelect allows to perform univariate feature selection with a configurable strategy. This allows to select the best univariate selection strategy with hyper-parameter search estimator.

    problemType = config['problem']
    
    targetcol   = config['target']['colname']
    
    selectionConfig = config['features']['Selection']
    criteria  = selectionConfig['critieria']
    threshold = selectionConfig['threshold']
    test      = selectionConfig['test']

    X = pddf.loc[:, pddf.columns != targetcol]
    y = pddf.loc[:, pddf.columns == targetcol]
    
    info("Feature data has "+getNcols(X, asStr=True)+" columns.", ind=6)
    
    if isRegression(problemType):
        if criteria == "top":
            if test == "f":
                Xnew = SelectKBest(f_regression, k=threshold).fit_transform(X, y)
            elif test == "info":
                Xnew = SelectKBest(mutual_info_regression, k=threshold).fit_transform(X, y)
            else:
                raise ValueError("Regression Test Criteria",test,"is unknown.")
        elif criteria == "percentile":
            if test == "f":
                Xnew = SelectPercentile(f_regression, k=threshold).fit_transform(X, y)
            elif test == "info":
                Xnew = SelectPercentile(mutual_info_regression, k=threshold).fit_transform(X, y)
            else:
                raise ValueError("Regression Test Criteria",test,"is unknown.")
        else:
            raise ValueError("Regression Selection Criteria",criteria,"is unknown.")
            
    if isClassification(problemType):
        if criteria == "top":
            if test == "chi2":
                Xnew = SelectKBest(chi2, k=threshold).fit_transform(X, y)
            elif test == "f":
                Xnew = SelectKBest(f_classif, k=threshold).fit_transform(X, y)
            elif test == "info":
                Xnew = SelectKBest(mutual_info_classif, k=threshold).fit_transform(X, y)
            else:
                raise ValueError("Classification Test Criteria",test,"is unknown.")
        elif criteria == "percentile":
            if test == "chi2":
                Xnew = SelectPercentile(chi2, k=threshold).fit_transform(X, y)
            elif test == "f":
                Xnew = SelectPercentile(f_classif, k=threshold).fit_transform(X, y)
            elif test == "info":
                Xnew = SelectPercentile(mutual_info_classif, k=threshold).fit_transform(X, y)
            else:
                raise ValueError("Classification Test Criteria",test,"is unknown.")
        else:
            raise ValueError("Classification Selection Criteria",criteria,"is unknown.")
            
    
    info("Feature data has "+getNcols(Xnew, asStr=True)+" columns.", ind=6)
    info("Joining new feature and target data.", ind=6)
    pddf = concat(Xnew, y)
    
    
    
def getCovarianceEstimates(X):
    info("Getting covariance estimates", ind=2)

    # fit a Minimum Covariance Determinant (MCD) robust estimator to data
    info("Getting Minimum Covariance Determinant", ind=4)
    robust_cov = MinCovDet().fit(X)

    # compare estimators learnt from the full data set with true parameters
    info("Getting Empirical Covariance", ind=4)
    emp_cov = EmpiricalCovariance().fit(X)
    
    return robust_cov, emp_cov