from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from ml_pipeline.regression.houseprices_dnn import DNNRegressorAdapter

REGRESSION_MODELS = {
    "dummy": DummyRegressor,
    "ridge": Ridge,
    "lasso": Lasso,
    "linear": ElasticNet,
    "knn": KNeighborsRegressor,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    # boosting
    "catboost": CatBoostRegressor,
    "lightgbm": LGBMRegressor,
    "xgboost": XGBRegressor,
    # DNN
    "dnn": DNNRegressorAdapter,
}
