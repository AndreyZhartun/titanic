import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score

from preprocessing import MyScaler
from config import config

class Linear():
    def __init__(self, df: pd.DataFrame, *, model: str) -> None:
        self.X = df.drop(columns=['Survived'])
        self.y = df.Survived

        if model == 'logreg':
            self.model = LogisticRegression(random_state=config.general.seed)
        elif model == 'ridge':
            self.model = RidgeClassifier(random_state=config.general.seed)
    
    def cv(self):
        n_folds = config.split.cv_n_folds

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.general.seed)
        losses = np.ndarray(n_folds)
        accuracy_list = np.ndarray(n_folds)
        
        for i, (train_index, test_index) in enumerate(kf.split(self.X)):
            train_X = self.X.iloc[train_index]

            scaler = MyScaler(
                num_cols=['Age', 'Fare'],
                mm_cols=['Pclass', 'SibSp', 'Parch', 'Family_Size', 'Age_group', 'Fare_Range']
            )

            train_X = scaler.fit_transform(train_X)
            train_X = train_X.to_numpy()

            train_y = self.y.iloc[train_index].to_numpy()

            fitted = self.model.fit(train_X, train_y)

            # test
            test_X = self.X.iloc[test_index]

            test_X = scaler.transform(test_X)
            test_X = test_X.to_numpy()

            y_pred = fitted.predict(test_X)
            y_true = self.y.iloc[test_index]
            
            losses[i] = mean_squared_error(y_true, y_pred)
            accuracy_list[i] = accuracy_score(y_true, y_pred)

        return (accuracy_list, accuracy_list.mean())

        
