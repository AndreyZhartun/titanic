import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureAdder:
    """
    Класс добавления базовых фичей
    """

    def __init__(self) -> None:
        pass

    def fit(self, _: pd.DataFrame):
        # nothing to fit
        return self

    def transform(self, input_df: pd.DataFrame):

        return input_df


class FeatureDropper:
    """
    Класс удаления фичей
    """

    def __init__(self, columns: list[str]) -> None:
        self.cols = columns

    def fit(self, _: pd.DataFrame):
        # nothing to fit
        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        df = df.drop(columns=self.cols)

        return df


class Imputer:
    """
    Класс замещения пустых значений фичей
    """

    def __init__(self, cols) -> None:
        self.cols = cols
        self.mean_values = {col: 0 for col in cols}

    def fit(self, df: pd.DataFrame):
        self.mean_values = {col: df[col].mean() for col in self.cols}

        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        # заполнение средним
        df.fillna(self.mean_values, inplace=True)

        return df


class CatEncoder:
    """
    Кодировка категориальных
    """

    def __init__(self) -> None:
        pass

    def fit(self, _: pd.DataFrame):
        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        df.replace({"Sex": {"male": 0, "female": 1}}, inplace=True)

        df["Sex"] = df["Sex"].astype("int")

        embarked_dummies = pd.get_dummies(df.Embarked, dtype=float)

        df["Embarked_C"] = embarked_dummies["C"]
        df["Embarked_S"] = embarked_dummies["S"]

        return df


class OneHotEncoder:
    def __init__(self, cols) -> None:
        self.cols = cols

    def fit(self, _: pd.DataFrame):
        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        df = pd.get_dummies(df, columns=self.cols, dtype=int)

        return self


class ContEncoder:
    """
    Кодировка непрерывных
    """

    def __init__(self) -> None:
        self.age_bins = []
        self.fare_bins = []
        pass

    def fit(self, df: pd.DataFrame):
        # запомнить бины из train, чтобы потом использовать на test
        age_bin_data = pd.cut(df["Age"], bins=5, retbins=True)
        # inf добавляется, чтобы данные из теста, выходящие за границы бинов, не были NaN
        self.age_bins = [-np.inf, *age_bin_data[1], np.inf]

        fare_bin_data = pd.cut(df["Fare"], bins=4, retbins=True)
        self.fare_bins = [-np.inf, *fare_bin_data[1], np.inf]

        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        # непрерывный Age в категориальный Age_group
        df["Age_Group"] = pd.cut(
            df["Age"], bins=self.age_bins, labels=range(0, len(self.age_bins) - 1)
        )

        df["Age_Group"] = df["Age_Group"].astype("int")

        # fare range
        df["Fare_Range"] = pd.cut(
            df["Fare"], bins=self.fare_bins, labels=range(0, len(self.fare_bins) - 1)
        )

        df["Fare_Range"] = df["Fare_Range"].astype("int")

        return df


class Scaler:
    """
    Стандартный и минмакс скейлер
    """

    def __init__(self, *, num_cols: list[str], mm_cols: list[str]) -> None:
        self.num_cols = num_cols
        self.mm_cols = mm_cols

        self.scaler = StandardScaler()
        self.mm_scaler = MinMaxScaler()

    def fit(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        # df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])
        self.scaler.fit(df[self.num_cols])

        self.mm_scaler.fit(df[self.mm_cols])

        return df

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        # хз почему скейлер дублирует колонки вместо того, чтобы заменять их
        # поэтому тут невероятные костыли
        df = df.drop(columns=[*self.mm_cols, *self.num_cols])

        df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        df[self.mm_cols] = self.mm_scaler.transform(input_df[self.mm_cols])

        return df


# Реестр, маппит "name" из конфига к классу препроцессора
REGRESSION_PREPROCESSORS: dict = {
    "feature_adder": FeatureAdder,
    "imputer": Imputer,
    "feature_dropper": FeatureDropper,
    "cat_encoder": CatEncoder,
    "cont_encoder": ContEncoder,
    "one_hot": OneHotEncoder,
    "scaler": Scaler,
}
