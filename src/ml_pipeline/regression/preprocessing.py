import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LogTransformer:
    """
    Класс преобразования фичей в log-нутые
    """

    def __init__(self, columns) -> None:
        self.cols = columns
        # self.mean_values = {col: 0 for col in cols}

    def fit(self, df: pd.DataFrame):
        # self.mean_values = {col: df[col].mean() for col in self.cols}

        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        # заполнение средним
        # df.fillna(self.mean_values, inplace=True)

        return df


# Реестр, маппит "name" из конфига к классу препроцессора
REGRESSION_PREPROCESSORS: dict = {
    "log_transformer": LogTransformer,
}
