import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessingTranformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        #self.get_feature_names_out = X.columns
        self.feature_names = X.columns
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        # convert date feature to datetime
        X_.date = pd.to_datetime(X_.date)
        #X_.year = X_.date.dt.year
        #X_.month = X_.date.dt.month
        #X_.week = X_.date.dt.week
        #X_.day_of_week = X_.date.dt.dayofweek
        return X_

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

