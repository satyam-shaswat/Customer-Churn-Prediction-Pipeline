import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def drop_customer_id(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the customerID column from the dataset.
    """
    return data.drop(columns=["customerID"])


def split_features_target(
    data: pd.DataFrame,
    target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features (X) and target (y).
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def encode_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all categorical columns.
    """
    categorical_columns = data.select_dtypes(include=["object"]).columns

    encoded_data = pd.get_dummies(
        data,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    return encoded_data


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split features and target into train and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):
    """
    Scale numerical features using StandardScaler.

    The scaler is fitted ONLY on the training data
    to prevent data leakage.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler

def save_processed_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a CSV file.
    """
    data.to_csv(file_path, index=False)


def load_processed_data(file_path: str) -> pd.DataFrame:
    """
    Load a processed CSV file.
    """
    return pd.read_csv(file_path)