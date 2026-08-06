# Demo: split_dataset Copilot example
# This small script demonstrates a function to split a pandas DataFrame
# into training and testing sets. It's intended as a Copilot playground.

import numpy as np
import pandas as pd


def split_dataset(dataset, test_ratio=0.20, random_seed=None):
    """Split a pandas DataFrame into (train, test) by a random mask.

    Args:
        dataset (pd.DataFrame): Input DataFrame to split.
        test_ratio (float): Fraction of rows to assign to the test set (0-1).
        random_seed (int|None): Optional seed for reproducible splits.

    Returns:
        tuple: (train_df, test_df)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate boolean mask for test rows
    test_mask = np.random.rand(len(dataset)) < test_ratio

    # Use boolean indexing on the DataFrame
    train_df = dataset.loc[~test_mask].reset_index(drop=True)
    test_df = dataset.loc[test_mask].reset_index(drop=True)

    return train_df, test_df


if __name__ == "__main__":
    # Quick demo dataset
    df = pd.DataFrame({
        "feature": range(100),
        "label": [x % 2 for x in range(100)],
    })

    train, test = split_dataset(df, test_ratio=0.2, random_seed=42)
    print(f"{len(train)} examples in training, {len(test)} examples in testing.")
    # Show a few rows to verify
    print(train.head())
