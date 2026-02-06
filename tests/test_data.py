import pandas as pd
from src.data.validate import validate_columns


def test_validate_columns_ok():
    df = pd.DataFrame({"a": [1], "b": [2]})
    validate_columns(df, ["a", "b"])


def test_validate_columns_missing():
    df = pd.DataFrame({"a": [1]})
    try:
        validate_columns(df, ["a", "b"])
        assert False, "Expected ValueError"
    except ValueError:
        assert True
