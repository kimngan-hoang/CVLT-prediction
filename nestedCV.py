from preprocess import *

#feature importance
from sklearn.inspection import permutation_importance
  
def grid_search(model, param_grid, score, inner_cv, X_train, y_train):
    '''
    Function for inner CV grid search
    
    Takes in initialized model, parameter grid, 
    performance metric (score, e.g., neg_mean_squared_error),
    number of cross-validation (cv), features (X), and output (y),
    
    Returns the best_model based on grid search (best_model),
    best performance metric (score) across all cv folds,
    and best set of hyperparameters
    '''
    search = GridSearchCV(estimator=model, 
                          param_grid = param_grid,
                          scoring = score,
                          cv = inner_cv,
                          n_jobs = 1,
                          refit = True)
    search.fit(X = X_train, y=y_train)
    best_model = search.best_estimator_
    best_score = abs(search.best_score_)
    best_param = search.best_params_
    return best_model, best_score, best_param


def performance(best_model, X, y):
    '''
    Calculates evaluation metrics using best-tuned model
    '''
    y_hat = best_model.predict(X)
    mse = mean_squared_error(y, y_hat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_hat)
    
    #Correlation between true vs predicted
    y_array = np.array(y).reshape(1,len(y))
    y_hat_array = pd.DataFrame(y_hat)
    y_hat_array = np.array(y_hat_array).reshape(1,len(y_hat))
    corr_true_pred = np.corrcoef(y_array, y_hat_array)[0,1]
    
    return rmse, r2, corr_true_pred

def feature_importance(best_model, X, y, fold_no, importance_df,
                       random_state = 42, scoring = 'neg_mean_squared_error'):
    
    # perform permutation importance
    results = permutation_importance(best_model, 
                                           X, y, 
                                           random_state = random_state, 
                                           scoring = scoring,
                                           n_repeats=5000)
    # get importance
    importance = results.importances_mean
    
    #Create df column for importance
    column_name = 'Feature Importance '+ str(fold_no-1)
    importance_df[column_name] = importance
    return importance_df
    
def nested_cv(feature_names, target, df, model, param_grid, score = 'neg_mean_squared_error', 
              inner_cv_num=5, outer_cv_num=10):
    
    #Empty list to save data
    outer_train_rmse = []
    outer_train_corelation = []
    outer_train_r2 = []

    outer_test_rmse = []
    outer_test_corelation = []
    outer_test_r2 = []

    importance_train_df = pd.DataFrame()
    importance_test_df = pd.DataFrame()

    # Feature Scaling
    scaler = MinMaxScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])

    # configure the outer cv procedure
    outer_cv = StratifiedKFold(n_splits=outer_cv_num, 
                               shuffle=True, random_state=0) 
    
    #fold counter
    fold_no = 1

    # Loop through each outer CV fold
    for train_index_outer, test_index_outer in outer_cv.split(df, df.grp):
        train_set = df.loc[train_index_outer,:]
        test_set = df.loc[test_index_outer,:]

        X_train = train_set[feature_names]
        y_train = train_set[target]
        X_test = test_set[feature_names]
        y_test = test_set[target]
        print("\n Results for outer loop fold ", fold_no)
        fold_no = fold_no+1
        
        best_model, best_score, best_param = grid_search(model = model, 
                                                        param_grid= param_grid,
                                                        score=score,
                                                        inner_cv = inner_cv_num,
                                                        X_train = X_train,
                                                        y_train = y_train)
        print('        Best MSE (across all inner validation folds):', best_score)
        print('        Best parameters:', best_param)
        
        #Outer Train set performance
        #This is to compare with performance on the outer test set and identify fitting issues
        rmse_train, r2_train, corr_true_pred_train = performance(best_model=best_model,
                                                                 X=X_train,
                                                                 y=y_train)
        outer_train_rmse.append(rmse_train)
        outer_train_r2.append(r2_train)
        outer_train_corelation.append(corr_true_pred_train) 
        
        print('\n        RMSE (on outer training set)', rmse_train)
        print('        R2 (on outer training set)', r2_train)
        print('        Correlation between Predicted and actual values (on outer training set)', corr_true_pred_train)
        
        ######## EVALUATE ON OUTER TEST SET ########
        rmse_test, r2_test, corr_true_pred_test = performance(best_model=best_model,
                                                              X=X_test,
                                                              y=y_test)
        
        outer_test_rmse.append(rmse_test)
        outer_test_r2.append(r2_test) 
        outer_test_corelation.append(corr_true_pred_test) 
        
        print('\n        RMSE (on outer test set)', rmse_test)
        print('        R2 (on outer test set)', r2_test)
        print('        Correlation between Predicted and actual values (on outer test set)', corr_true_pred_test)
        
        ######## Feature importance #########
        ### Training set   
        importance_train_df = feature_importance(best_model, X_train, y_train, fold_no, importance_train_df)
        ### Test set
        importance_test_df = feature_importance(best_model, X_train, y_train, fold_no, importance_test_df)
        
    # Print evaluation metrics across all outer loop folds
    print('\n    Average performance across all outer test sets:')
    print('        RMSE %.2f +/- %.2f'% (np.mean(outer_test_rmse), np.std(outer_test_rmse)))
    print('        R2 %.2f +/- %.2f'% (np.mean(outer_test_r2), np.std(outer_test_r2)))
    print('        Correlation %.2f +/- %.2f'% (np.mean(outer_test_corelation), np.std(outer_test_corelation)))
    
    return np.array(outer_test_rmse), np.array(outer_test_r2), np.array(outer_test_corelation), importance_train_df, importance_test_df


#test codes 
if __name__ == '__main__':
    #Import dataset
    data = pd.read_csv("C:/Users/kimng/Desktop/ML - Age, Hipp, CVLT/CVLTHippocampus.csv")
    data = data.reset_index(drop=True)
    columns = ['CVLT_Imm_Total', 'CVLT_DelR_SD_Free', 'CVLT_DelR_LD_Free',
            'Age','Sex', 'EduYears', 'Smoker', 'High_BP', 'COMT', 'BDNF2', 'ApoE4',
            'LH_Total', 'RH_Total',
           'L_HH_Total', 'R_HH_Total', 'L_HB_Total', 'R_HB_Total', 'L_HT_Total', 'R_HT_Total',
           'L_DG_Total', 'R_DG_Total',
           'L_CA_Total', 'R_CA_Total',
           'L_Sub_Total', 'R_Sub_Total',
           'L_HH_CA', 'R_HH_CA', 'L_HB_CA', 'R_HB_CA', 'L_HT_CA', 'R_HT_CA', 
           'L_HH_DG', 'R_HH_DG', 'L_HB_DG', 'R_HB_DG', 'L_HT_DG', 'R_HT_DG',
           'L_HH_Sub', 'R_HH_Sub', 'L_HB_Sub', 'R_HB_Sub', 'L_HT_Sub', 'R_HT_Sub']
    #preprocess
    df = data[columns].dropna(inplace = True)

    #Initialize Regressor
    model = ElasticNet()

    #Set up parameter grid
    param_grid = [{'alpha': np.arange(0.01, 2.01, 0.01)}]
    feature_names = ['Age','Sex', 'EduYears', 'COMT', 'BDNF2', 'ApoE4', 'Smoker', 'High_BP'
                 ]
    target = ['CVLT_Imm_Total']
    
    _, _, _, _, _, = nested_cv(feature_names=feature_names,
                               target=target,
                               df=df,
                               model = model,
                               param_grid=param_grid
                               )
