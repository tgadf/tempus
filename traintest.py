
# coding: utf-8

from pandas import Series
from datetime import datetime as dt
from numpy import prod
from sklearn.base import ClassifierMixin,RegressorMixin
from xgboost.sklearn import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


###########################################################################
#
# Train Model
#
###########################################################################
def trainEstimator(name, estimator, X_train, y_train, debug = False):
    """
    Train a scikit-learn estimator
    
    Inputs:
      > name: string
      > estimator: a scikit-learn estimator (Regressor or Classifier)
      > X_train: the training data (DataFrame)
      > y_train: the response data (DataFrame or Series)
      > debug: (False) set to True for print statements
      
    Output: scikit-learn estimator (although not needed since fit results are stored in estimator)
    """
    if debug:
        startTime = dt.now()
        print("Training {0} on {1} rows and {2} features.".format(name, X_train.shape[0], X_train.shape[1]))
    estimator.fit(X_train, y_train)
    if debug:
        endTime = dt.now()
        delta = endTime-startTime
        totaltime = delta.seconds
    
        if totaltime > 60:
            units = "min"
            totaltime /= 60.0
        else:
            units = "sec"
        
        totaltime = round(totaltime,1)
        print("Time to train {0} is {1} {2}.".format(name, totaltime, units))
        
    return estimator



###########################################################################
#
# Predict Model
#
###########################################################################
def predictEstimator(name, estimator, X_test, y_test = None, y_scaler = None, debug = False):
    """
    Predict response from a scikit-learn estimator
    
    Inputs:
      > name: string
      > estimator: a scikit-learn estimator (Regressor or Classifier)
      > X_test: the test data (DataFrame)
      > y_test: (None) the test response data (DataFrame or Series) (for formating)
      > y_scaler: (None) if the response requires transformation
      > debug: (False) set to True for print statements
      
    Output: Dictionary {"probs": class probabilities (if Classifer),
                        "labels": class labels (if Classifier),
                        "values": response values (if Regressor}
    """
    
    if debug:
        startTime = dt.now()
        print("Predicting {0} on {1} rows and {2} features.".format(name, X_test.shape[0], X_test.shape[1]))
        
    if isinstance(estimator, ClassifierMixin):
        probs  = getProbabilities(name, estimator, X_test)
        probs.index = X_test.index
        labels = getPredictions(name, estimator, X_test)
        labels.index = X_test.index
    else:
        probs  = None
        labels = None
    
    if isinstance(estimator, RegressorMixin):
        values = getPredictions(name, estimator, X_test)
        values.index = X_test.index
    else:
        values = None

    ## Invert predicted data
    if y_scaler is not None:
        values = invertTransform(values, y_scaler)
        values.index = X_test.index

    retval = {"probs": probs, "labels": labels, "values": values}
    
    if debug:
        endTime = dt.now()
        delta = endTime-startTime
        totaltime = delta.seconds
    
        if totaltime > 60:
            units = "min"
            totaltime /= 60.0
        else:
            units = "sec"
        
        totaltime = round(totaltime,1)
        print("Time to predict {0} is {1} {2}.".format(name, totaltime, units))

    return retval 


def getProbabilities(name, estimator, X_test, debug = False):
    """
    Predict class probabilities from a trained estimator
    
    Inputs:
      > name: string
      > estimator: a scikit-learn estimator (Regressor or Classifier)
      > X_test: the test data (DataFrame)
      > debug: (False) set to True for print statements
      
    Output:
      > pandas.Series of class probabilities
    """

    if debug:
        print("  Computing target probabilities for {0}".format(name))
    probs  = estimator.predict_proba(X_test)[:,1]
    probs  = Series(data=probs, name="predicted")
    #probs.index = y_test.index
    return probs

def getPredictions(name, estimator, X_test, debug = False):
    """
    Predict class labels (Classifier) or response (Regressor) from a trained estimator
    
    Inputs:
      > name: string
      > estimator: a scikit-learn estimator (Regressor or Classifier)
      > X_test: the test data (DataFrame)
      > debug: (False) set to True for print statements
      
    Output:
      > pandas.Series of class labels/response
    """
    
    if debug:
        print("  Computing target predictions for {0}".format(name))
    preds  = estimator.predict(X_test)
    preds  = Series(data=preds, name="predicted")
    #preds.index = y_test.index
    return preds



###########################################################################
#
# Tune Estimator
#
###########################################################################
def tuneEstimator(modelname, estimator, X_train, y_train, config, debug = False):
    """
    Hyperparameter tune a scikit-learn estimator
    
    Inputs:
      > modelname: string
      > estimator: a scikit-learn estimator (Regressor or Classifier)
      > X_train: the training data (DataFrame)
      > y_train: the response data (DataFrame or Series)
      > config: Dictionary {"grid": Dictionary of parameters to scan (example below),
                            "njobs": number of parallel threads (jobs),
                            "type": "grid" or "random",
                            "iter": number of random iterations to test (10 if not specified)}
      > debug: (False) set to True for print statements

        Example grid for xgboost: grid = {'min_child_weight': [10, 100, 1000], 'max_depth': [2, 6, 10], 'learning_rate': 0.1, 'gamma': 0.3}
        
    Output:
      > pandas.Series of class labels/response
    """
    
    if debug:
        print("Tuning a {0} estimator".format(modelname))
        verbose=1
    else:
        verbose=0
    
    if estimator is None:
        print("There is no estimator with parameters information.")
        return {"estimator": None, "params": None, "cv": None}


    if config.get('grid') is not None:
        grid = config['grid']
    else:
        if isinstance(estimator, [XGBRegressor, XGBClassifier]):
            grid = {'min_child_weight': [10, 100, 1000], 'max_depth': [2, 6, 10], 'learning_rate': 0.1, 'gamma': 0.3}
        else:
            print("No grid for {0}".format(modelname))
            return {"estimator": estimator, "params": estimator.get_params(), "cv": None}


    scorers = []
    if isinstance(estimator, ClassifierMixin):
        scorers = ["accuracy", "average_precision", "f1", "f1_micro",
                   "f1_macro", "f1_weighted", "f1_samples", "neg_log_loss",
                   "precision", "recall", "roc_auc"]
        scorer = "roc_auc"
    

    if isinstance(estimator, RegressorMixin):
        scorers = ["explained_variance", "neg_mean_absolute_error",
                   "neg_mean_squared_error", "neg_mean_squared_log_error",
                   "neg_median_absolute_error", "r2"]
        scorer = "neg_mean_absolute_error"

    if scorer not in scorers:
        raise ValueError("Scorer {0} is not allowed".format(scorer))

    searchType = config.get('type')
    if searchType is None:
        searchType = "random"
    if config.get('njobs') is not None:
        njobs = config['njobs']
    else:
        njobs = 2
        
    if searchType == "grid":
        tuneEstimator = GridSearchCV(estimator, param_grid=grid, cv=2,
                                     scoring=scorer, verbose=verbose, n_jobs=njobs)
    elif searchType == "random":
        n_iter_search = config.get('iter')
        if n_iter_search is None:
            nMax  = 10
            n_iter_search = min(nMax, prod([len(x) for x in config['grid'].values()]))         
        tuneEstimator = RandomizedSearchCV(estimator, param_distributions=grid,
                                           cv=2, n_iter=n_iter_search,
                                           verbose=verbose, n_jobs=njobs,
                                           return_train_score=True)
    else:
        raise ValueError("Search type {0} is not allowed".format(searchType))

    if debug:
        print("Running {0} parameter search".format(searchType))
    tuneEstimator.fit(X_train, y_train)
    bestEstimator = tuneEstimator.best_estimator_        
    bestScore     = tuneEstimator.best_score_
    bestParams    = tuneEstimator.best_params_
    cvResults     = tuneEstimator.cv_results_
    cvScores      = cvResults['mean_test_score']
    fitTimes      = cvResults['mean_fit_time']

    if debug:
        print("Tested {0} Parameter Sets".format(len(fitTimes)))
        print("CV Fit Time Info (Mean,Std): ({0} , {1})".format(round(fitTimes.mean(),1), round(fitTimes.std(),1)))
        print("Best Score                 : {0}".format(round(bestScore, 3)))
        print("Worst Score                : {0}".format(round(min(cvScores), 3)))
        print("CV Test Scores (Mean,Std)  : ({0} , {1})".format(round(cvScores.mean(),1), round(cvScores.std(),1)))
        print("Best Parameters")
        for paramName, paramVal in bestParams.items():
            print("Param: {0} = {1}".format(paramName, paramVal))    

    return {"estimator": bestEstimator, "params": bestParams, "cv": cvResults}

