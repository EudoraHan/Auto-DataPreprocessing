
"""
Created on Wed Jul 17 13:42:10 2019
@author: Yun Han


Automatic Datapreprocessing Function
1. Data format 
2. Missing Value
3. Outlier Dectect


"""

### Data Preprocessing

### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Part 1. Data formatting

"""

## Label Encoding for String
from sklearn.preprocessing import LabelEncoder

def labelencode(data):
    labelencoder_X = LabelEncoder()
    data = labelencoder_X.fit_transform(data)
    integer_mapping = {l: i for i, l in enumerate(labelencoder_X.classes_)}
    
    return data, integer_mapping

"""
Part 2. Use different method to deal with missing value

"""
# =============================================================================
# ### Detect missing value
# df.info() # the overview information for the dataframe
# df.describe() # basic stats
# df.isnull().sum() # the number of rows with NaN for each column
# df.notnull().sum() # the number of rows without NaN for each column
# 
# =============================================================================


def missing_value(data, method):
    if method == 'delete':
        return data.dropna(inplace=True)
    
    elif method == '0 impute':
        return data.fillna(0, inplace=True) 
    
    elif method == 'mean':
        return data.fillna(data.mean(), inplace=True)
    
    elif method == 'median':
        return data.fillna(data.median(), inplace=True)
    
    elif method == 'ffill':
        return data.fillna(method='ffill', inplace = True)
    
    elif method == 'bfill':
        return data.fillna(method='bfill', inplace = True)
    
    elif method == 'interpolation':
        return data.interpolate()

# =============================================================================
# ### KNN for imputation
#         
# from sklearn.neighbors import KNeighborsClassifier
# # construct X matrix
# X = df.iloc[:, :-1].values
# column_new = ['RevolvingUtilizationOfUnsecuredLines', 'age', 
#               'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 
#               'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
#               'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 
#               'NumberOfDependents']
# X = pd.DataFrame(data=X, columns = column_new)
# 
# # select the rows with missing values as testing data
# idx_with_nan = X.age.isnull()
# X_with_nan = X[idx_with_nan]
# 
# # select the rows without missing values as training data
# X_no_nan = X[-idx_with_nan]
# 
# # drop name column, set age as target variable and train the model, 
# clf = KNeighborsClassifier(3, weights='distance')
# clf.fit(X_no_nan[['RevolvingUtilizationOfUnsecuredLines',
#                   'NumberOfTime30-59DaysPastDueNotWorse', 
#                   'NumberOfOpenCreditLinesAndLoans', 
#                   'NumberOfTimes90DaysLate', 
#                   'NumberRealEstateLoansOrLines', 
#                   'NumberOfTime60-89DaysPastDueNotWorse']], X_no_nan['age'])
# 
# # impute the NA value
# x_imputed = clf.predict(X_with_nan[['RevolvingUtilizationOfUnsecuredLines',
#                   'NumberOfTime30-59DaysPastDueNotWorse', 
#                   'NumberOfOpenCreditLinesAndLoans', 
#                   'NumberOfTimes90DaysLate', 
#                   'NumberRealEstateLoansOrLines', 
#                   'NumberOfTime60-89DaysPastDueNotWorse']])
# X_with_imputed = X.copy()
# X_with_imputed.loc[idx_with_nan,'age'] = x_imputed.reshape(-1)
# 
# =============================================================================

"""
Part 3. Anomaly detection/ Outliers detection
https://www.zhihu.com/question/38066650

"""

# =============================================================================
# ### Handle Outliers
# import seaborn as sns
# 
# # Simulate data
# sns.set_style('whitegrid')
# sns.distplot(df.DebtRatio)
# sns.distplot(df.MonthlyIncome)
# sns.boxplot(df.age,orient='v')
# 
# 
# =============================================================================

##### auto function for outlier

from scipy.stats import skew
    
# 1. Numeric Outlier: define a function remove outlier using IQR: one dimension
def iqr_outlier(data):
    lq,uq=np.percentile(data,[25,75])
    lower_l=lq - 1.5*(uq-lq)
    upper_l=uq + 1.5*(uq-lq)
    
    # calculate the ratio of outliers
    ratio = (len(data[(data > upper_l)])+len(data[(data < lower_l)]))/len(data)
    # if ratio is large, we might replace the outlier with boundary value.
    if ratio > 0.1:
        
        return data
    
    elif ratio > 0.05:
        data[data < lower_l] = lower_l
        data[data > upper_l] = upper_l
        print ("%d upper is:", upper_l)
        print (data,"lower is:", lower_l)
        return data
        
    else:
        return data[(data >=lower_l)&(data<=upper_l)]
    
    
# 2. Z-score：one dimension or low dimension
def z_score_outlier(data):
    
    threshold=3
    mean_y = np.mean(data)
    stdev_y = np.std(data)
    z_scores = [(y - mean_y) / stdev_y for y in data]
    
    return data[np.abs(z_scores) < threshold]

"""
Auto function for outlier： 
combine the first two function

"""

def outlier(data):
    skewness = skew(data)    
    if skewness > 1:
        remove_outlier = iqr_outlier(data)
        
    else:
        remove_outlier = z_score_outlier(data)
    
    return remove_outlier

# =============================================================================
# ### Isolation Forest: one dimension or high dimension） 
#     
# # https://zhuanlan.zhihu.com/p/27777266
#     
# from sklearn.ensemble import IsolationForest
# import pandas as pd
# 
# 
# clf = IsolationForest(max_samples=100, random_state=42)
# clf.fit(train)
# y_pred = clf.predict(train)
# y_pred = [1 if x == -1 else 0 for x in y_pred]
# y_pred.count(0) # 94714
# y_pred.count(1) # 10524  
# 
# 
# =============================================================================


"""
Part 4. Auto Datapreprocessing Function 
1. a single variable
2. Whole dataset

"""
# For a single variable

def preprocessing(i, data, type, method):    
    
    """
    判断：
    
    数值型变量中如果含有string，需要剔除
    分类型变量中如果是string，用encoding的方式
    
    """
    
    if type == 'numeric':    
        if data[i].dtype == 'O':
            data[i] = pd.to_numeric(data[i], errors='coerce')
            
        missing_value(data[i], method)
        clean_data = outlier(data[i])        
        return clean_data
    
    elif type == 'categorical':
        missing_value(data[i], method)
        pre_index = data[i].index
        
        if data[i].dtype == 'O':
            data, dictionary = labelencode(data[i])        
            data = pd.Series(data, name = i)
            data.index = pre_index
            clean_data = outlier(data)    
        else:
            clean_data = outlier(data[i])  
        return clean_data


# For a whole dataset

def clean_all(df, categorical, method_cate, method_numeric):    
    for i in df.columns:
        if i not in categorical: 
            clean = preprocessing(i, df, 'numeric', method_numeric)
            if len(clean) < len(df):
                df = pd.merge(clean, df, left_index=True,right_index=True, how='left',suffixes=('_x', '_delete')) # left_on, right_on
            else:
                df = pd.merge(clean, df, left_index=True,right_index=True, how='right',suffixes=('_x', '_delete')) # left_on, right_on
      
        else:
            clean = preprocessing(i, df, 'categorical', method_cate)
            if len(clean) < len(df):
                df = pd.merge(clean, df, left_index=True,right_index=True, how='left',suffixes=('_x', '_delete')) # left_on, right_on
            else:
                df = pd.merge(clean, df, left_index=True,right_index=True, how='right',suffixes=('_x', '_delete')) # left_on, right_on
    
    for name in df.columns:
        if "_delete"  in name:
            df = df.drop([name], axis=1)
    
    return df



"""
Part 5. Dataset TEST
1. Titanic
2. givemecredit

"""
# 1. Titanic
tit = pd.read_csv('titanic.csv')
tit = tit.drop(['name','cabin'], axis=1)

cat = ['survived', 'pclass', 'sex', 'embarked']        
after_Clean = clean_all(tit, cat, 'delete', 'median')        

# 2. givemecredit
credit = pd.read_csv('givemecredit.csv')
cat = []        
after_Clean = clean_all(credit, cat, 'delete', 'median')        


# =============================================================================
#### Step Test Code
# clean = preprocessing('survived', tit, 'categorical', 'delete')
# tit = pd.merge(tit, clean, left_index=True,right_index=True, how='left') # left_on, right_on
# clean = preprocessing('pclass', tit, 'categorical', 'delete')
# clean = preprocessing('embarked', tit, 'categorical', 'delete')
# 
# =============================================================================

