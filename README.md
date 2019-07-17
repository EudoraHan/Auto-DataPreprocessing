# Auto-DataPreprocessing

## Automatic Datapreprocessing Function
## Data format 
1. Data cleaning: Delete the string in numeric column 
2. Label Encoding for string in categorical variable

## Missing Value
1. Elimination
drop the rows with any NaN, equal to df.dropna(axis=0, how='any')
2. Imputation 
2.1 fill 0 to all missing values
2.2 fill missing values with mean/median (NUMERIC)
2.3 carry the backward value to the missing value (Both NUMERIC and OBJECT)
2.4 carry the forward value to the missing value
2.5 Use interpolation to fill the missing value 
2.6 KNN for imputation

## Outlier Dectect
1. IQR: define a function remove outlier using IQR(one dimension)
2. Z-score：one dimension or low dimension
3. Isolation Forest: one dimension or high dimension）
