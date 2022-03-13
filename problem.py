import os
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import rampwf as rw

problem_title = "Auto Judge Challenge"

_prediction_label_names = [0, 1]
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names
)

# Force our own score
custom_workflow = rw.utils.importing.import_module_from_source(
        os.path.join(os.getcwd(), 'custom_workflow.py'),
        'custom_workflow',
        sanitize=True
)
workflow = custom_workflow.Classifier()

# Force our own score
custom_score = rw.utils.importing.import_module_from_source(
        os.path.join(os.getcwd(), 'custom_score.py'),
        'custom_score',
        sanitize=True
)

score_types = [custom_score.F1(name='F1-score')]

_target_column_name = 'winning_index'
_ignore_column_names = [
    'conclusion',
    'votes',
    'majority_vote',
    'minority_vote',
    'winning_party',
    'disposition',
    'decision_type',
    'unconstitutionality'
]


def _get_data(path=".", f_name="train"):
    data = pd.read_pickle(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values - 1
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path="."):
    return _get_data(path, "train.pkl")


def get_test_data(path="."):
    return _get_data(path, "test.pkl")


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)
