import pandas as pd
from sklearn.model_selection import train_test_split
from modelhelper import *

# Read the data
X_full = pd.read_csv('data/input/train.csv', index_col='Id')
X_test_full = pd.read_csv('data/input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
# X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
#                                                                 train_size=0.8, test_size=0.2,
#                                                                 random_state=0)



# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
# categorical_cols = [cname for cname in X_train_full.columns if
#                     X_train_full[cname].nunique() < 10 and 
#                     X_train_full[cname].dtype == "object"]

# Select numerical columns
# numerical_cols = [cname for cname in X_train_full.columns if 
#                 X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only 
# keep_cols = categorical_cols + numerical_cols
# X_train = X_train_full[keep_cols].copy()
# X_valid = X_valid_full[keep_cols].copy()
# X_test = X_test_full[keep_cols].copy()


# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
# categorical_cols = [cname for cname in X_full.columns if
#                     X_full[cname].nunique() < 10 and 
#                     X_full[cname].dtype == "object"]

# Select numerical columns
# numerical_cols = [cname for cname in X_full.columns if 
#                 X_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only 

categorical_cols , numerical_cols = keepCols(X_full, drop_thresh=.9)
keep_cols = categorical_cols + numerical_cols

X = X_full[keep_cols].copy()
X_test = X_test_full[keep_cols].copy()


# Pipeline ###################################
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

the_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', model)
                        ])
the_pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', model2)
                        ])

kValidate(the_pipeline, X, y, cv = 5)
kValidate(the_pipeline2, X, y, cv = 5)

# clf = tryModel(clf, X_train, y_train, X_valid, y_valid)
# clf = tryModel(clf2, X_train, y_train, X_valid, y_valid)


# Predict and submit
the_pipeline2.fit(X,y)
preds_test = the_pipeline2.predict(X_test) 

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('data/output/submission.csv', index=False)


