import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess(input_df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ['Sex', 'Pclass', 'SibSp', 'Embarked', 'Parch']
    numeric_cols = ['Age', 'Fare']

    # TODO process other cols
    df = input_df[[*cat_cols, *numeric_cols, 'Survived']]

    # TODO handle N/A better
    df = df.dropna()

    if 'Sex' in df.columns:
        df['is_female'] = (df.Sex == 'female').astype('int')
        df = df.drop(columns=['Sex'])

    if 'Embarked' in df.columns:
        embarked_dummies = pd.get_dummies(df.Embarked, dtype=float)

        df['Embarked_C'] = embarked_dummies['C']
        df['Embarked_S'] = embarked_dummies['S']

        df = df.drop(columns=['Embarked'])

    scaler = StandardScaler()

    # TODO scale after train/test split
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols]) # type: ignore

    cols_to_minmax_scale = cat_cols[:]
    cols_to_minmax_scale.remove('Sex')
    cols_to_minmax_scale.remove('Embarked')

    mm_scaler = MinMaxScaler()

    df[cols_to_minmax_scale] = mm_scaler.fit_transform(df[cols_to_minmax_scale]) # type: ignore
    print(df)
    return df