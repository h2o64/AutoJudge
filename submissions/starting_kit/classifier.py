from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# Custom transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import array
import scipy.sparse as sp


# Select the columns of the pandas dataframe
class SelectColumnsTransformer():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, X, **transform_params):
        cpy_df = X[self.columns].copy()
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# Silent the warning about unknown classes on MultiLabelBinarize
class CustomMultiLabelBinarizer(MultiLabelBinarizer):
    def _transform(self, y, class_mapping):
        indices = array.array("i")
        indptr = array.array("i", [0])
        for labels in y:
            index = set()
            for label in labels:
                try:
                    index.add(class_mapping[label])
                except KeyError:
                    pass
            indices.extend(index)
            indptr.append(len(indices))
        data = np.ones(len(indices), dtype=int)
        return sp.csr_matrix(
            (data, indices, indptr),
            shape=(len(indptr) - 1, len(class_mapping))
        )


# TF-IDF/Multilabel transform the text inputs
class ColumnsTextVectorizer():
    def __init__(self, columns_text=None, columns_multilabel=None):
        self.columns_text = columns_text
        self.columns_multilabel = columns_multilabel
        self.vectorizers = {
            c: TfidfVectorizer() for c in columns_text
        }
        for c in self.columns_multilabel:
            self.vectorizers[c] = CustomMultiLabelBinarizer()

    def transform(self, X, **transform_params):
        # Transform texts and multi labels
        texts = []
        for c in self.columns_text + self.columns_multilabel:
            X_tmp = self.vectorizers[c].transform(X[c])
            if c in self.columns_text:
                X_tmp = X_tmp.todense()
            texts.append(X_tmp)
        # Transform non-text
        col_text_multilabel = self.columns_text + self.columns_multilabel
        non_texts = X[[c for c in X.columns if c not in col_text_multilabel]]
        non_texts = non_texts.values
        # Concatenate everything
        return np.asarray(np.concatenate(texts + [non_texts], axis=1))

    def fit(self, X, y=None, **fit_params):
        # Fit the vectorizers
        for c in self.columns_text + self.columns_multilabel:
            self.vectorizers[c].fit(X[c])
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# Classifier
class Classifier(BaseEstimator):
    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, X, Y):
        input_columns = [
            'first_party_type',
            'second_party_type',
            'facts_of_the_case',
            'province_name',
            'heard_by',
            'decided_by'
        ]
        # Build the pipeline
        self.pipeline = Pipeline([
            (
                'columns_selection',
                SelectColumnsTransformer(columns=list(input_columns))
            ),
            ('vectorization', ColumnsTextVectorizer(
                columns_text=[
                    'first_party_type',
                    'second_party_type',
                    'facts_of_the_case',
                    'province_name'
                ],
                columns_multilabel=['heard_by', 'decided_by']

            )),
            ('classifier', self.model)
        ])

        # Fit the classifier
        self.pipeline = self.pipeline.fit(X, Y)

    def predict_proba(self, X):
        # Perform the prediction
        y_pred = self.pipeline.predict(X)
        # One-hot encode it
        y_pred = y_pred.astype(bool)
        ret = np.zeros((y_pred.shape[0], 2))
        ret[y_pred, 1] = 1
        ret[~y_pred, 0] = 1
        return ret
