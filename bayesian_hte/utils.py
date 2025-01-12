"Utility functions"

import pandas as pd


def check_binary(x: pd.Series) -> None:
    """
    Check whether input is binary, contains both 0 and 1.

    Parameters
    ----------
    x : pandas.Series

    Returns
    -------
    bool
    """
    is_binary = set(x.astype(int)) == {0, 1}
    if not is_binary:
        raise ValueError("Input is not binary.")
