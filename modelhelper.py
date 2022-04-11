
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

def keepCols(X, drop_thresh=.8):
    
    perc_missing = X.isnull().sum()/X.shape[0]
    drop_cols = perc_missing[perc_missing > drop_thresh].index.tolist()
    
    X.drop(drop_cols, axis=1, inplace=True)

    categorical_cols = [cname for cname in X.columns if
                    X[cname].nunique() < 10 and 
                    X[cname].dtype == "object"]

    numerical_cols = [cname for cname in X.columns if 
                    X[cname].dtype in ['int64', 'float64']]
    return (categorical_cols, numerical_cols)


def kValidate(the_pipeline, X, y, cv = 5):
    # Multiply by -1 since sklearn calculates *negative* MAE
    rmse_scorer = make_scorer(mean_squared_log_error, squared=False)
    scores = cross_val_score(the_pipeline, X, y,
                                  cv=cv,
                                  scoring=rmse_scorer )
                                #   scoring='neg_mean_absolute_error')
    # print("Average MAE score:", scores.mean())
    print("Average RMSE score:", scores.mean())
    

# Bundle preprocessing and modeling code in a pipeline
def tryModel(the_pipeline, X_train, y_train, X_valid, y_valid):
    # Preprocessing of training data, fit model 
    the_pipeline.fit(X_train, y_train)
    # Preprocessing of validation data, get predictions
    preds = the_pipeline.predict(X_valid)
    print('MAE:', mean_absolute_error(y_valid, preds))
    return the_pipeline