from xgboost import XGBRegressor

print(X.isnull().sum()/X.shape[0])




X_train.Alley.head()



X_train2, X_valid2 = X_train2.align(X_valid2, join='left', axis=1)
# X_train2, X_test2 = X_train2.align(X_test2, join='left', axis=1)

print(X_train2.isnull().sum()/X_train.shape[0])



my_model_1 = XGBRegressor()
# Fit the model
my_model_1.fit(X_train2, y_train)

predictions_1 = my_model_1.predict(X_valid2)