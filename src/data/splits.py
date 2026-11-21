import pandas as pd


def chronological_split(
    series: pd.Series,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
):
    """
    Split a time series chronologically into
    training, validation, and testing partitions.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("Expected a pandas Series.")

    if not 0 < train_fraction < 1:
        raise ValueError("Invalid training fraction.")

    if not 0 < validation_fraction < 1:
        raise ValueError("Invalid validation fraction.")

    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training + validation must be below 1.")

    n = len(series)

    train_end = int(n * train_fraction)

    validation_end = int(
        n * (train_fraction + validation_fraction)
    )

    train = series.iloc[:train_end].copy()

    validation = series.iloc[
        train_end:validation_end
    ].copy()

    test = series.iloc[validation_end:].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One or more partitions are empty.")

    return train, validation, test
