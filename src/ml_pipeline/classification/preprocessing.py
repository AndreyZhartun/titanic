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
        df = input_df.copy()

        df["Initial"] = df["Name"].str.extract("([A-Za-z]+)\\.")

        df.replace(
            {
                "Initial": {
                    "Mlle": "Miss",
                    "Mme": "Miss",
                    "Ms": "Miss",
                    "Dr": "Mr",
                    "Major": "Mr",
                    "Lady": "Mrs",
                    "Countess": "Mrs",
                    # "Jonkheer": "Other",
                    # "Col": "Other",
                    # "Rev": "Other",
                    "Capt": "Mr",
                    "Sir": "Mr",
                    "Don": "Mr",
                }
            },
            inplace=True,
        )

        # все другие титулы просто Other
        df.loc[~df["Initial"].isin(["Mr", "Miss", "Mrs"]), "Initial"] = "Other"

        # размер семьи
        df["Family_Size"] = df["Parch"] + df["SibSp"]
        # alone - 1 если нет семьи
        df["Alone"] = 0
        df.loc[df.Family_Size == 0, "Alone"] = 1

        # 4 и больше родственников
        df["More_Than_4_relatives"] = 0
        df.loc[df.Family_Size >= 4, "More_Than_4_relatives"] = 1

        # порядок билета
        df["Ticket_Type"] = df["Ticket"].apply(lambda x: x[0:3])
        df["Ticket_Type"] = df["Ticket_Type"].astype("category")
        df["Ticket_Type"] = df["Ticket_Type"].cat.codes

        return df


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

    def __init__(self) -> None:
        self.mean_ages_by_initials = {}
        self.Embarked_mode = ""
        self.Fare_mean = 0

    def fit(self, df: pd.DataFrame):
        # заполнить Age средним по Initial
        self.mean_ages_by_initials = df.groupby("Initial").Age.mean().round()

        # получить моду для Embarked
        self.Embarked_mode = df.Embarked.mode()[0]

        # получить среднее для  Fare
        self.Fare_mean = df.Fare.mean()

        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        for initial in list(self.mean_ages_by_initials.keys()):
            df.loc[(df.Age.isnull()) & (df.Initial == initial), "Age"] = (
                self.mean_ages_by_initials[initial]
            )

        # заполнение модой
        df.fillna({"Embarked": self.Embarked_mode}, inplace=True)

        # заполнение средним
        df.fillna({"Fare": self.Fare_mean}, inplace=True)

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
        self.mm_scaler.fit(df[self.mm_cols])

        return df

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        # df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        # хз почему скейлер дублирует колонки вместо того, чтобы заменять их
        # поэтому тут невероятные костыли
        df = df.drop(columns=self.mm_cols)

        df[self.mm_cols] = self.mm_scaler.transform(input_df[self.mm_cols])

        return df


# Реестр, маппит "name" из конфига к классу препроцессора
CLASSIFICATION_PREPROCESSORS: dict = {
    "feature_adder": FeatureAdder,
    "imputer": Imputer,
    "feature_dropper": FeatureDropper,
    "cat_encoder": CatEncoder,
    "cont_encoder": ContEncoder,
    "scaler": Scaler,
}
