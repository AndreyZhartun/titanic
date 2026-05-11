import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if 'Sex' in df.columns:
        df['is_female'] = (df.Sex == 'female').astype('int')
        df = df.drop(columns=['Sex'])

    if 'Embarked' in df.columns:
        # S, потому что S наибольшее количество в данных
        df.fillna({'Embarked': 'S'},inplace=True)

        embarked_dummies = pd.get_dummies(df.Embarked, dtype=float)

        df['Embarked_C'] = embarked_dummies['C']
        df['Embarked_S'] = embarked_dummies['S']

        df = df.drop(columns=['Embarked'])

    df = fill_missing_age(df)

    # groups from 0 to 4 for Age (continous Age to categorical Age_group)
    df['Age_group'] = pd.cut(df['Age'], bins=5, labels=range(0, 5))

    # family size
    df['Family_Size'] = df['Parch'] + df['SibSp']
    df['Alone'] = 0
    df.loc[df.Family_Size == 0, 'Alone'] = 1

    # fare range
    df['Fare_Range'] = pd.qcut(df['Fare'], 4, labels=range(0, 4))

    # drop cols
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Initial'])
    
    return df


def fill_missing_age(input_df: pd.DataFrame) -> pd.DataFrame:

    df = input_df.copy()

    if 'Age' in df.columns:
        for i in df:
            df['Initial'] = df['Name'].str.extract('([A-Za-z]+)\\.')
        
        df.replace({
            'Initial': {
                'Mlle': 'Miss',
                'Mme': 'Miss',
                'Ms': 'Miss',
                'Dr': 'Other',
                'Major': 'Mr',
                'Lady': 'Mrs',
                'Countess': 'Mrs',
                'Jonkheer': 'Other',
                'Col': 'Other',
                'Rev': 'Other',
                'Capt': 'Mr',
                'Sir': 'Mr',
                'Don': 'Mr'
            }},
            inplace=True
        )

        mean_ages_by_initials = df.groupby('Initial').Age.mean().round()

        for initial in list(mean_ages_by_initials.keys()):
            df.loc[
                (df.Age.isnull()) & (df.Initial == initial),
                'Age'
            ] = mean_ages_by_initials[initial]

        assert bool(df.Age.isnull().any()) is False

    return df

def scale(input_df: pd.DataFrame):
    df = input_df.copy()

    scaler = StandardScaler()

    numeric_cols = ['Age', 'Fare']

    # TODO scale after train/test split
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    cols_to_minmax_scale = ['Pclass', 'SibSp', 'Parch', 'Family_Size', 'Age_group', 'Fare_Range']
    mm_scaler = MinMaxScaler()
    df[cols_to_minmax_scale] = mm_scaler.fit_transform(df[cols_to_minmax_scale])

    return (df, scaler, mm_scaler)

class MyScaler():
    def __init__(self, *, num_cols: list[str], mm_cols: list[str]) -> None:
        self.num_cols = num_cols
        self.mm_cols = mm_cols

        self.scaler = StandardScaler()
        self.mm_scaler = MinMaxScaler()

    def fit_transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])
        df[self.mm_cols] = self.mm_scaler.fit_transform(df[self.mm_cols])

        return df
    
    def transform(self, input_df: pd.DataFrame):
        df = input_df.copy()

        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        df[self.mm_cols] = self.mm_scaler.transform(df[self.mm_cols])

        return df