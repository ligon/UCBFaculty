import pandas as pd

from ucbfaculty import transformations


def test_fullname_builds_title_case():
    df = pd.DataFrame({"FirstName": ["c.", "JANE"], "LastName": ["blanton", "DOE"]})
    result = df.apply(transformations.fullname, axis=1)
    assert list(result) == ["C. Blanton", "Jane Doe"]


def test_fullname_handles_missing():
    df = pd.DataFrame({"FirstName": ["", None], "LastName": ["Smith", "Jones"]})
    result = df.apply(transformations.fullname, axis=1)
    assert pd.isna(result.iloc[1])
    assert result.iloc[0] == "Smith"


def test_title_handles_scalar_and_series():
    assert transformations.title("c. BLANTON") == "C. Blanton"
    series = pd.Series(["FOO", "bar"])
    transformed = transformations.title(series)
    assert list(transformed) == ["Foo", "Bar"]

