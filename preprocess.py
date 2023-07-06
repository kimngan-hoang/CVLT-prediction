import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet


#Function for categorise dataframe
def categorise(row):
    if row['AgeGroup'] == 1 and row['Sex'] == 0:
        return 1
    elif row['AgeGroup'] == 1 and row['Sex'] == 1:
        return 2
    elif row['AgeGroup'] == 2 and row['Sex'] == 0:
        return 3
    elif row['AgeGroup'] == 2 and row['Sex'] == 1:
        return 4
    elif row['AgeGroup'] == 3 and row['Sex'] == 0:
        return 5
    elif row['AgeGroup'] == 3 and row['Sex'] == 1:
        return 6
    elif row['AgeGroup'] == 4 and row['Sex'] == 0:
        return 7
    elif row['AgeGroup'] == 4 and row['Sex'] == 1:
        return 8
    elif row['AgeGroup'] == 5 and row['Sex'] == 0:
        return 9
    elif row['AgeGroup'] == 5 and row['Sex'] == 1:
        return 10
    elif row['AgeGroup'] == 6 and row['Sex'] == 0:
        return 11
    elif row['AgeGroup'] == 6 and row['Sex'] == 1:
        return 12
    elif row['AgeGroup'] == 7 and row['Sex'] == 0:
        return 13
    elif row['AgeGroup'] == 7 and row['Sex'] == 1:
        return 14
    elif row['AgeGroup'] == 8 and row['Sex'] == 0:
        return 15
    elif row['AgeGroup'] == 8 and row['Sex'] == 1:
        return 16

def preprocessing(df):
    #Bin Age into groups
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    labels = [1,2,3,4,5,6,7,8]
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    df = df.reset_index(drop=True)
    #Apply categories to dataframe
    df['grp'] = df.apply(lambda row: categorise(row), axis=1)
    
    return df
