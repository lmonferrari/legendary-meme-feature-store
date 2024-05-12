import numpy as np
import pandas as pd


def create_feature_store():
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    n_features_group_1 = 2
    n_features_group_2 = 10
    n_random = n_features - (n_features_group_1 + n_features_group_2)

    features_group_1 = np.random.randn(n_samples, n_features_group_1)
    features_group_2 = np.dot(
        features_group_1, np.random.rand(n_features_group_1, n_features_group_2)
    )

    features_group_3 = np.random.randn(n_samples, n_random)

    X = np.hstack([features_group_1, features_group_2, features_group_3])
    y = (features_group_1[:, 0] + features_group_1[:, 1] > 0).astype(int)

    df_features = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df_features["target"] = y

    return df_features
