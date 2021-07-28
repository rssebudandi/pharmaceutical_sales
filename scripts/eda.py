#Import neccessary libraries
import logging

import mlflow
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn import preprocessing

#Read in the different datasets
#Read in store

logging.basicConfig(level=logging.INFO)
store=pd.read_csv('../data/store.csv')


#Read in test data
test=store_data=pd.read_csv('../data/test.csv')

#Read in train data
train=store_data=pd.read_csv('../data/train.csv')

#Read in sample submission
sample_submission=store_data=pd.read_csv('../data/sample_submission.csv')

train.shape, test.shape, store.shape
#logging.info("The shape of {train}, and {test} datasets ".format(train.shape,test.shape))

def joinDF(df1,df2):
    df1.Store.nunique() == df2.Store.nunique()
    df_combined = df1.merge(df2, how='left', left_on=df1.Store, right_on=df2.Store)
    df_combined.drop(['key_0', 'Store_y'], axis=1, inplace=True)
    df_combined = df_combined.rename(columns={'Store_x': 'Store'})
    return df_combined




if __name__ == '__main__':
    mlflow.set_experiment(experiment_name='EDA')
    store_columns = store.columns
    mlflow.log_param("Store columns", store_columns)
    train_columns = train.columns
    mlflow.log_param("Train columns", train_columns)
    test_columns = test.columns
    mlflow.log_param("Test columns", test_columns)

    df_combined=joinDF(train,store)
    #mlflow.log_param("Combined store and train sets", df_combined)
    Summary_start = round(df_combined.describe().T, 2)
    #mlflow.log_param("Sumarry statistics", Summary_start)

    #create more columns
    df_combined.Date = pd.to_datetime(df_combined.Date)
    df_combined['Day'] = df_combined.Date.dt.day
    df_combined['Month'] = df_combined.Date.dt.month
    df_combined['Year'] = df_combined.Date.dt.year
    df_combined
    #mlflow.log_param("Extended", df_combined)

    #mlflow.log_artifact("ElasticNet-paths.png")



