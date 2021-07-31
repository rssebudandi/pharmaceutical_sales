#Import neccessary libraries
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

#Read in the different datasets
#Read in store
store=pd.read_csv('../data/store.csv')
#Read in test data
test=store_data=pd.read_csv('../data/test.csv')

#Read in train data
train=store_data=pd.read_csv('../data/train.csv')

#Read in sample submission
sample_submission=store_data=pd.read_csv('../data/sample_submission.csv')


def add_colums(df):
    df.Date = pd.to_datetime(df.Date)
    df['Day'] = df.Date.dt.day
    df['Month'] = df.Date.dt.month
    df['Year'] = df.Date.dt.year
    df['Weekday'] = df.Date.dt.weekday
    df['Month_start'] = df.Date.dt.is_month_start
    df['Month_end'] = df.Date.dt.is_month_end

    return df


# encode data
def encode_data(df):
    month_start_encoder = preprocessing.LabelEncoder()
    month_end_encoder = preprocessing.LabelEncoder()
    date_encoder = preprocessing.LabelEncoder()
    state_hol_encoder = preprocessing.LabelEncoder()
    day_encoder = preprocessing.LabelEncoder()
    month_encoder = preprocessing.LabelEncoder()
    year_encoder = preprocessing.LabelEncoder()
    weekday_encoder = preprocessing.LabelEncoder()
    open_encoder = preprocessing.LabelEncoder()

    df['Month_start'] = month_start_encoder.fit_transform(df['Month_start'])
    df['Month_end'] = month_end_encoder.fit_transform(df['Month_end'])
    df['Date'] = date_encoder.fit_transform(df['Date'])
    df['Day'] = day_encoder.fit_transform(df['Day'])
    df['Month'] = month_encoder.fit_transform(df['Month'])
    df['Year'] = year_encoder.fit_transform(df['Year'])
    df['Weekday'] = weekday_encoder.fit_transform(df['Weekday'])
    df['Open'] = open_encoder.fit_transform(df['Open'])
    return df
#drop column
def drop_col(df,col):
    df=df[df.columns[~df.columns.isin([col])]]
    return df

#select feature columns
# select feature columns
def select_features(df):
    if 'Sales' in df.columns:
        feature_col = ["DayOfWeek", "Date", "Open", "Promo", "SchoolHoliday", "Day", "Month"]
        features_X = df[feature_col]
        target_y = train_clean["Sales"]

        return features_X, target_y
    else:
        feature_col = ["DayOfWeek", "Date", "Open", "Promo", "SchoolHoliday", "Day", "Month"]
        features_X = df[feature_col]
        return features_X


if __name__ == '__main__':
    train_modified=add_colums(train)
    print(train_modified)
    train_encoded = encode_data(train_modified)
    print(train_encoded)
    col = 'StateHoliday'
    train_clean=drop_col(train_encoded, col)
    print(train_clean)
    train_features,target=select_features(train_clean)
    print(train_features)
    print(target)
    #Testing
    # test data
    test_modified = add_colums(test)
    # print(test_modified)
    test_encoded = encode_data(test_modified)
    # print(train_encoded)
    col = 'StateHoliday'
    test_clean = drop_col(test_encoded, col)
    # print(train_clean)
    test_features = select_features(test_clean)
    # print(train_features)
    test_features

    #using a sklearn pipeline
    # random forest regressor pipe
    piperf = Pipeline([
        ('scalar', StandardScaler()),

        ('random_forest', RandomForestRegressor(max_depth=2, random_state=2))
    ])
    piperf.fit(train_features,target)
    print(piperf.predict(test_features))



