
# coding: utf-8

from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, MinMaxScaler, RobustScaler
from pandas import DataFrame, Series
from numpy import reshape

def transformData(transtype, X_data, y_data):
    """
    Transform data using scikit-learn transformers
    
    Inputs:
      > transtype: string of the type of transformation (minmax,quantile, or robust). If not recognized there won't be any transformation
      > X_data: the feature data
      > y_data: the target data
      
    Output:
      > Dictionary: {"X_data": X_data, "y_data": y_data, "X_scaler": X_scaler, "y_scaler": y_scaler}
    """
    y_scaler = None
    X_scaler = None

    if transtype == "minmax":
        print("Scaling Using MinMaxScaler")
        y_scaler = MinMaxScaler()
        X_scaler = MinMaxScaler()
    elif transtype == "quantile":
        print("Scaling Using QuantileTransformer")
        y_scaler = QuantileTransformer()
        X_scaler = QuantileTransformer()
    elif transtype == "robust":
        print("Scaling Using RobustScaler")
        y_scaler = RobustScaler()
        X_scaler = RobustScaler()
    else:
        print("No idea about {0}".format(transtype))

    if X_scaler is not None:
        ## Scale X
        print("Transforming X data")
        Xscaled  = X_scaler.fit_transform(Xdata)
        X_data = DataFrame(Xscaled, columns=features.columns)

    if y_scaler is not None:
        ## Scale Y
        print("Transforming y data")
        y = reshape(y_data, (-1, 1))
        yscaled = y_scaler.fit_transform(y)
        y_data = Series(yscaled.ravel(), name=targetcol)
        
    retval = {"X_data": X_data, "y_data": y_data, "X_scaler": X_scaler, "y_scaler": y_scaler}
    return retval


def invertTransform(transformed_data, transformer = None):
    """
    Invert the transformed data
    
    Inputs:
      > transformed_data: The data that has been transformed
      > transformer (None by default): If not None, this is the scikit-learn transform returned by the transformData function
      
    Output:
      > The inverted data (if transformer is passed) or the original data (if transformed is None)
    """
    if transformer is not None:
        print("Inverting data")
        if isinstance(transformed_data, ndarray):
            if transformed_data.ndim == 1:
                print("  Reshaping transformed data")
                transformed_data = reshape(transformed_data, (-1, 1))
                inverted_data = transformer.inverse_transform(transformed_data)
        if isinstance(transformed_data, DataFrame):
            columns = transformed_data.columns
            inverted_data = transformer.inverse_transform(transformed_data)
            inverted_data = DataFrame(inverted_data, columns=columns)
        if isinstance(transformed_data, Series):
            name = transformed_data.name
            transformed_data = reshape(transformed_data, (-1, 1))
            inverted_data = transformer.inverse_transform(transformed_data)
            inverted_data = Series(inverted_data.ravel(), name=name)
        return inverted_data
    return transformed_data