import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline


class FeatureAdder:
    """
    Class to add basic features
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

        df.loc[~df["Initial"].isin(["Mr", "Miss", "Mrs"]), "Initial"] = "Other"

        # family size
        df["Family_Size"] = df["Parch"] + df["SibSp"]
        # alone - 1 if no family
        df["Alone"] = 0
        df.loc[df.Family_Size == 0, "Alone"] = 1

        return df


class FeatureDropper:
    """
    Class to drop unneeded features
    """

    def __init__(self) -> None:
        pass

    def fit(self, _: pd.DataFrame):
        # nothing to fit
        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        # drop name, ticket, cabin, initial because we cannot extract any more info
        # drop embarked - it was one-hot encoded
        # drop age and fare because we are using bins instead of continous features
        df = df.drop(
            columns=["Name", "Ticket", "Cabin", "Initial", "Embarked", "Age", "Fare"]
        )

        return df


class Imputer:
    def __init__(self) -> None:
        self.mean_ages_by_initials = {}
        self.Embarked_mode = ""

    def fit(self, df: pd.DataFrame):
        # fill Age using average for Initial
        self.mean_ages_by_initials = df.groupby("Initial").Age.mean().round()

        # get mode for Embarked
        self.Embarked_mode = df.Embarked.mode()[0]

        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        for initial in list(self.mean_ages_by_initials.keys()):
            df.loc[(df.Age.isnull()) & (df.Initial == initial), "Age"] = (
                self.mean_ages_by_initials[initial]
            )

        # fill with mode
        df.fillna({"Embarked": self.Embarked_mode}, inplace=True)

        return df


class CatEncoder:
    def __init__(self) -> None:
        pass

    def fit(self, _: pd.DataFrame):
        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        df.replace({"Sex": {"male": 0, "female": 1}}, inplace=True)

        embarked_dummies = pd.get_dummies(df.Embarked, dtype=float)

        df["Embarked_C"] = embarked_dummies["C"]
        df["Embarked_S"] = embarked_dummies["S"]

        return df


class ContEncoder:
    def __init__(self) -> None:
        self.age_bins = []
        self.fare_bins = []
        pass

    def fit(self, df: pd.DataFrame):
        # remember bins from train to use on test
        age_bin_data = pd.cut(df["Age"], bins=5, retbins=True)
        self.age_bins = [-np.inf, *age_bin_data[1], np.inf]

        fare_bin_data = pd.cut(df["Fare"], bins=4, retbins=True)
        self.fare_bins = [-np.inf, *fare_bin_data[1], np.inf]

        return self

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        # groups from 0 to 4 for Age (continous Age to categorical Age_group)
        df["Age_Group"] = pd.cut(
            df["Age"], bins=self.age_bins, labels=range(0, len(self.age_bins) - 1)
        )

        # fare range
        df["Fare_Range"] = pd.cut(
            df["Fare"], bins=self.fare_bins, labels=range(0, len(self.fare_bins) - 1)
        )

        return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    # df = add_features(df)

    # df = fill_na(df)

    # df = encode_cats(df)

    # df = encode_continuous(df)

    # df = drop_features(df)

    return df


def fill_missing_age(input_df: pd.DataFrame) -> pd.DataFrame:

    df = input_df.copy()

    mean_ages_by_initials = df.groupby("Initial").Age.mean().round()

    for initial in list(mean_ages_by_initials.keys()):
        df.loc[(df.Age.isnull()) & (df.Initial == initial), "Age"] = (
            mean_ages_by_initials[initial]
        )

    assert bool(df.Age.isnull().any()) is False

    return df


class MyScaler:
    def __init__(self, *, num_cols: list[str], mm_cols: list[str]) -> None:
        self.num_cols = num_cols
        self.mm_cols = mm_cols

        self.scaler = StandardScaler()
        self.mm_scaler = MinMaxScaler()

    def fit_transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        # df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])
        df[self.mm_cols] = self.mm_scaler.fit_transform(df[self.mm_cols])

        return df

    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        # df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        df[self.mm_cols] = self.mm_scaler.transform(df[self.mm_cols])

        return df


# Registry
# maps the "name" string from config.py to the transformer class
TRANSFORMER_REGISTRY: dict = {
    "feature_adder": FeatureAdder,
    "imputer": Imputer,
    "feature_dropper": FeatureDropper,
    "cat_encoder": CatEncoder,
    "cont_encoder": ContEncoder,
}
