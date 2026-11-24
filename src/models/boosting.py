"""The histogram gradient-boosting regressor selected in notebook 07, rebuilt for any set of predictors."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def boosting_regressor(numeric, categorical, parameters):
    """Pipeline with notebook 07's preprocessing (median imputation, ordinal codes for rare-grouped categories).

    ``parameters`` are the estimator parameters saved by notebook 07; ``categorical_features`` is recomputed so that
    it points to the categorical columns of this predictor set.
    """
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), list(numeric)),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       min_frequency=25, max_categories=80)),
        ]), list(categorical)),
    ], remainder="drop", sparse_threshold=0)
    estimator_parameters = {**parameters,
                            "categorical_features": list(range(len(numeric), len(numeric) + len(categorical)))}
    return Pipeline([("preprocessor", preprocessor),
                     ("estimator", HistGradientBoostingRegressor(**estimator_parameters))])
