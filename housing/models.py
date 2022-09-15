from housing import Pipeline, data_holder, make_column_transformer
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.linear_model import ElasticNet, Lasso, SGDRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.kernel_ridge import KernelRidge


def cross_validation_error(estimator):
    """Calculates the negative RMSE for the given estimator using 5-fold cross validation
    """
    data = data_holder.get_data()
    X = data.drop('price', axis=1)
    score = cross_val_score(estimator, X,
                            data['price'], scoring='neg_root_mean_squared_error')
    return [score.mean(), score.std()]


def create_xgb():
    return xgb.XGBRegressor(colsample_bytree=0.5, gamma=0.05,
                            learning_rate=0.05, max_depth=4,
                            min_child_weight=1.5, n_estimators=2000,
                            reg_alpha=0.4, reg_lambda=0.8,
                            subsample=0.5,
                            random_state=1)


def create_elastic_net():
    return ElasticNet(alpha=0.001, l1_ratio=.0009, random_state=1)


def create_kernel_ridge():
    return KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


def create_lasso():
    return Lasso(alpha=0.00005, random_state=1)


def create_sdg():
    return SGDRegressor(loss='huber', alpha=0.00001,
                        epsilon=0.5, power_t=0.001, eta0=0.4, learning_rate='adaptive')


def grid_search(estimator, param_grid):
    """Performs GridSearchCV for the estimator in the param_grid and
    returns the best score and the best params
      --------
      Parameters:
        estimator: estimator obj
          The estimator
        para_grid: dict or list of dictionaries
          grid to perform search
    """
    data = data_holder.get_data()
    CV_en = GridSearchCV(estimator=estimator, param_grid=param_grid,
                         cv=5, scoring='neg_root_mean_squared_error')
    X = data.drop('price', axis=1)
    results = CV_en.fit(X, data['price'])
    return [results.best_score_, results.best_params_]


def get_non_ohe_columns():
    """Gets a list with the columns that haven't been one hot encoded
    """
    data = data_holder.get_data()
    one_hot_columns = [x for x in data.columns if 'onehotencoder' in x]
    return list((set(data.columns)-set(one_hot_columns))-set(['price']))


def make_standard_scaler_transformer(columns=None):
    """Makes a ColumTransformer object with StandardScaler() over the columns given
     --------
        Parameters:
          columns: [str]
            the name of the columns
    """
    if (columns):
        return make_column_transformer((StandardScaler(), columns), remainder='passthrough')
    else:
        return StandardScaler()
