from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from ml_pipeline.classification.dnn import DNNAdapter

REGRESSION_MODELS = {
    "dummy": DummyRegressor,
    "linear": ElasticNet,
    "knn": KNeighborsRegressor,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    # boosting
    "catboost": CatBoostRegressor,
    "lightgbm": LGBMRegressor,
    "xgboost": XGBRegressor,
    # DNN
    "dnn": DNNAdapter,
}
