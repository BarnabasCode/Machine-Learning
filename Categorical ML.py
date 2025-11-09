print("Hi")

import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('melb_data.csv')


# Separate target from predictors
y = data.Price
x = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)


# Drop columns with missing values (simplest approach)
cols_with_miss = [col for col in x_train.columns if x_train[col].isnull().any()]

x_train.drop(cols_with_miss, axis=1, inplace = True)
x_val.drop(cols_with_miss, axis=1, inplace = True)


# "Cardinality" means the number of unique values in a column
low_cardinality_cols = [cname for cname in x_train.columns if x_train[cname].nunique() < 10 and x_train[cname].dtype == "object"]
print(low_cardinality_cols)

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

numericalcol = [numcol for numcol in x_train.columns if x_train[numcol].dtype in ['int64', 'float64']]
print(numericalcol)

# Select numerical columns

# Keep selected columns only
