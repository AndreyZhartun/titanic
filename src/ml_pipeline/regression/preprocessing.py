"""
Классы препроцессоров, специфические для регрессии house prices
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LogTransformer:
    """
    Класс преобразования фичей в log-нутые
    """

    def __init__(self, columns) -> None:
        self.columns = columns

    def fit(self, _: pd.DataFrame):

        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        # логарифмирование фич
        for col in self.columns:
            df[col] = np.log(df[col])

        return df


# Реестр, маппит "name" из конфига к классу препроцессора
REGRESSION_PREPROCESSORS: dict = {
    "log_transformer": LogTransformer,
}
