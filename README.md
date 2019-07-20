# Auto-DataPreprocessing

## Automatic Datapreprocessing Function
### Data format 
* Data cleaning: Delete the string in numeric column 
* Label Encoding for string in categorical variable

### Missing Value
* Elimination
drop the rows with any NaN, equal to df.dropna(axis=0, how='any')
* Imputation 

1. fill 0 to all missing values
2. fill missing values with mean/median (NUMERIC)
3. carry the backward value to the missing value (Both NUMERIC and OBJECT)
4. carry the forward value to the missing value
5. Use interpolation to fill the missing value 
6. KNN for imputation

### Outlier Dectect
* IQR: define a function remove outlier using IQR(one dimension)
* Z-score：one dimension or low dimension
* Isolation Forest: one dimension or high dimension）

### Author
Yun Han
