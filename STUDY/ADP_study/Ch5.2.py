from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

def get_linear_reg_eval(model_name, parmas, x, y, verbose=True, return_coeff=True) : 
    coeff_df = pd.DataFrame()
    for param in params : 
        if model_name == 'Ridge' : model = Ridge(alpha=param)
        elif model_name == 'Lasso':model=Lasso(alpha=param)
        elif model_name == 'ElasticNet' : model = ElasticNet(alpha=param, l1_ratio=0.7)
    neg_mse_scores = cross_val_score(lr, x, y, scoring='neg_mean_squared_error', cv=5)
    rmse_scores = np.sqrt(-1*neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print('alpha가',param,'일 때 5폴드 세트의 평균 RMSE :',avg_rams)