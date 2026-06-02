"""
Классы препроцессоров, специфические для титаника
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TitanicFeatureAdder:
    """
    Класс добавления фичей, найденных в ходе EDA
    """

    def __init__(self) -> None:
        pass

    def fit(self, _: pd.DataFrame):
        # nothing to fit
        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        df["Initial"] = df["Name"].str.extract("([A-Za-z]+)\\.")

        # все титулы объединяются в группы по самым ближайшим
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


class TitanicAgeImputer:
    """
    Класс замещения пустых значений Age
    """

    def __init__(self) -> None:
        self.mean_ages_by_initials = {}

    def fit(self, df: pd.DataFrame):
        # заполнить Age средним по Initial
        self.mean_ages_by_initials = df.groupby("Initial").Age.mean().round()

        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        for initial in list(self.mean_ages_by_initials.keys()):
            df.loc[(df.Age.isnull()) & (df.Initial == initial), "Age"] = (
                self.mean_ages_by_initials[initial]
            )

        return df


# реестр, маппит "name" из конфига к классу препроцессора
TITANIC_PREPROCESSORS: dict = {
    "titanic_feature_adder": TitanicFeatureAdder,
    "titanic_age_imputer": TitanicAgeImputer,
}
