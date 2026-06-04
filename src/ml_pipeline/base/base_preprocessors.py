"""
Самые базовые классы препроцессоров, которые могут быть использованы для любых датасетов
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BaseColumnDropper:
    """
    Удаляет фичи в списке columns
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, _: pd.DataFrame):
        return self

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return input_df.drop(columns=self.columns)


class BaseMeanImputer:
    """
    Заполняет пропуски средним значением по фичам columns
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self._means: dict = {}

    def fit(self, df: pd.DataFrame):
        self._means = {col: df[col].mean() for col in self.columns}

        return self

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df.fillna(self._means, inplace=True)

        return df


class BaseModeImputer:
    """
    Заполняет пропуски модой по фичам columns
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self._modes: dict = {}

    def fit(self, df: pd.DataFrame):
        self._modes = {col: df[col].mode()[0] for col in self.columns}

        return self

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df.fillna(self._modes, inplace=True)

        return df


class BaseMappingEncoder:
    """
    Кодирует значения колонки columns через явно заданный словарь mapping {old_value: new_value}.
    Для бинарных переменных и любых номинальных фичей с известным маппингом
    """

    def __init__(
        self,
        column: str,
        mapping: dict,
        # если указан, привести результат к этому типу
        dtype=None,
    ) -> None:
        self.column = column
        self.mapping = mapping
        self.dtype = dtype

    def fit(self, _: pd.DataFrame):
        return self

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        df[self.column] = df[self.column].replace(self.mapping)

        if self.dtype is not None:
            df[self.column] = df[self.column].astype(self.dtype)

        return df


class BaseOneHotEncoder:
    """
    One-hot кодировка фичей columns через pd.get_dummies.
    Запоминает колонки при fit, чтобы тест имел ту же структуру
    """

    def __init__(
        self,
        columns: list[str],
        # удалять ли исходные колонки
        drop_original: bool = True,
        # удалять ли первую колонку
        drop_first: bool = True,
        dtype=float,
        # словарь {колонка: префикс}. Если None, используется название колонки
        prefix: dict | None = None,
    ) -> None:
        self.columns = columns
        self.drop_original = drop_original
        self.drop_first = drop_first
        self.dtype = dtype
        self.prefix = prefix or {col: col for col in columns}
        self._encoded_cols: list[str] = []

    def fit(self, df: pd.DataFrame):
        dummy_df = pd.get_dummies(
            df[self.columns], dtype=self.dtype, prefix=self.prefix
        )

        if self.drop_first:
            # дропнуть первую one-hot-колонку для каждой фичи
            cols_to_drop = [
                next(
                    c for c in dummy_df.columns if c.startswith(f"{self.prefix[col]}_")
                )
                for col in self.columns
            ]

            dummy_df = dummy_df.drop(columns=cols_to_drop)

        self._encoded_cols = dummy_df.columns.tolist()

        return self

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        dummy_df = pd.get_dummies(
            df[self.columns], dtype=self.dtype, prefix=self.prefix
        )

        # привести к колонкам, увиденным при fit
        dummy_df = dummy_df.reindex(
            columns=self._encoded_cols,
            # добавить нули для отсутствующих
            fill_value=0,
        )

        # убрать исходные
        if self.drop_original:
            df = df.drop(columns=self.columns)

        return pd.concat([df, dummy_df], axis=1)


class BaseBinEncoder:
    """
    Переводит непрерывные колонки в категориальные через pd.cut.
    Бины запоминаются при fit, на тест применяются те же границы.
    Крайние бины расширяются до ±inf, чтобы тест не давал NaN
    """

    def __init__(
        self,
        # словарь {колонка: количество_бинов}
        columns_bins: dict[str, int],
        # суффикс для новых колонок
        suffix: str = "_bin",
        # удалять ли исходные колонки
        drop_original: bool = False,
    ) -> None:
        self.columns_bins = columns_bins
        self.suffix = suffix
        self.drop_original = drop_original
        self._bin_edges: dict[str, list] = {}

    def fit(self, df: pd.DataFrame):
        # для каждой фичи запомнить бины из train, чтобы потом использовать на test
        for col, n_bins in self.columns_bins.items():
            _, edges = pd.cut(
                df[col],
                bins=n_bins,
                # возвращает bin-ы, чтобы потом передать их в cut еще раз
                retbins=True,
            )
            # inf добавляется, чтобы данные из теста, выходящие за границы бинов, не были NaN
            self._bin_edges[col] = [-np.inf, *edges[1:-1], np.inf]

        return self

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        # для каждой фичи преобразовать ее в категориальную
        for col, edges in self._bin_edges.items():
            n_labels = len(edges) - 1

            new_col = f"{col}{self.suffix}"

            df[new_col] = pd.cut(df[col], bins=edges, labels=range(n_labels))

            df[new_col] = df[new_col].astype(int)

        if self.drop_original:
            df = df.drop(columns=list(self.columns_bins.keys()))

        return df


class BaseScaler:
    """
    Масштабирует фичи через StandardScaler (standard_cols) или MinMaxScaler (minmax_cols)
    """

    def __init__(
        self,
        standard_cols: list[str] | None = None,
        minmax_cols: list[str] | None = None,
    ) -> None:
        self.standard_cols = standard_cols or []
        self.minmax_cols = minmax_cols or []

        self._standard_scaler = StandardScaler() if self.standard_cols else None
        self._minmax_scaler = MinMaxScaler() if self.minmax_cols else None

    def fit(self, df: pd.DataFrame):
        if self._standard_scaler:
            self._standard_scaler.fit(df[self.standard_cols])

        if self._minmax_scaler:
            self._minmax_scaler.fit(df[self.minmax_cols])

        return self

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        if self._standard_scaler:

            scaled = self._standard_scaler.transform(df[self.standard_cols])

            scaled = pd.DataFrame(scaled, columns=self.standard_cols, index=df.index)

            for col in scaled:
                df[col] = scaled[col]

        if self._minmax_scaler:
            scaled = self._minmax_scaler.transform(df[self.minmax_cols])

            scaled = pd.DataFrame(scaled, columns=self.minmax_cols, index=df.index)

            for col in scaled:
                df[col] = scaled[col]

        return df


# реестр базовых препроцессоров, маппит "name" из конфига к классу препроцессора
# может быть дополнен препроцессорами для конкретного датасета
# конвенция: ключи начинаются с base_
BASE_PREPROCESSORS: dict = {
    "base_col_dropper": BaseColumnDropper,
    "base_mean_imputer": BaseMeanImputer,
    "base_mode_imputer": BaseModeImputer,
    "base_mapper": BaseMappingEncoder,
    "base_onehot": BaseOneHotEncoder,
    "base_bin_encoder": BaseBinEncoder,
    "base_scaler": BaseScaler,
}
