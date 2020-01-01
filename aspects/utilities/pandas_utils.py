import numpy as np
import pandas as pd

def assert_columns(df: pd.DataFrame, *columns: str, **columns_with_types):
    available_columns = set(df.columns)
    expected_columns = columns + tuple(columns_with_types.keys())
    difference = set(expected_columns) - set(available_columns)
    assert difference == set(), f"Expected columns {difference} are missing, available {available_columns}"

    if columns_with_types is not None:
        available_columns_with_types = dict(zip(df.columns, df.dtypes))
        for column, type in columns_with_types.items():
            got = available_columns_with_types[column]
            expected = np.dtype(type)
            assert got == expected, f'Column {column} has inappropriate type, expected {expected}, got {got}'
