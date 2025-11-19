import hashlib
import pandas as pd


# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.

def stable_bucket(key: None, training_frac: float) -> str:
    '''
    Labels an id with train or test depending on the hashed value

    Parameters
    ----------
    key : None - id value
    training_frac : float - Fraction to use for training, by default 0.8

    Returns
    -------
    str
        label containing train/test split based on IDs.

    '''
    h = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 16**8
    return "train" if v < training_frac else "test"

def create_sample_split(df: pd.DataFrame, id_column: str | None = None, training_frac: float=0.8) -> pd.DataFrame:
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    # TODO: Create a new column called sample with values train and test
    # TODO: use haslib based on the ID to create the bins based on the hash produced
    df["sample"] = df[id_column].apply(lambda x: stable_bucket(x, training_frac))
    return df
