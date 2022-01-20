import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessingTranformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = pd.DataFrame(X.copy())
        # convert date feature to datetime
        X_['date'] = pd.to_datetime(X_.date)
        X_['year'] = X_.date.dt.year
        X_['month'] = X_.date.dt.month
        X_['sin_month'] = np.sin(2*np.pi*X_.month/12)
        X_['cos_month'] = np.cos(2*np.pi*X_.month/12)
        #X_['week'] = X_.date.dt.isocalendar().week
        X_['day_of_week'] = X_.date.dt.dayofweek
        X_['sin_day_of_week'] = np.sin(2*np.pi*X_.day_of_week/6)
        X_['cos_day_of_week'] = np.cos(2*np.pi*X_.day_of_week/6)
        X_.drop(columns=['date','month','day_of_week'], inplace=True)
        return X_

    def get_feature_names_out(self, input_features=None):
        input_features = ['year','sin_month','cos_month', 'sin_day_of_week', 'cos_day_of_week']
        #input_features += [col for col in ['year','sin_month','cos_month', 'day_of_week'] if col not in input_features]
        return np.asarray(input_features, dtype=object)

