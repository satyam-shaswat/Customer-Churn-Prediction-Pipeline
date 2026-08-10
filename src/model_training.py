import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression( x_train, y_train, random_state=42):

    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000
    )

    model.fit(x_train, y_train)

    return model


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 100,
    random_state: int = 42
):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    n_estimators : int
        Number of decision trees.

    random_state : int
        Seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Trained Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model
from sklearn.model_selection import GridSearchCV


def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest using GridSearchCV.
    """

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def save_model(model, file_path: str):
    """
    Save a trained machine learning model.
    """
    joblib.dump(model, file_path)


def load_model(file_path: str):
    """
    Load a trained machine learning model.
    """
    return joblib.load(file_path)