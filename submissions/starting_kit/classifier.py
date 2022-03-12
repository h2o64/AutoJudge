from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, X, Y):
        input_columns = ['first_party_type', 'second_party_type', 'facts_of_the_case', 'province_name','heard_by','decided_by']
        # Build the pipeline
        self.pipeline = Pipeline([
            ('columns_selection', SelectColumnsTransformer(columns=list(input_columns))),
            ('vectorization', ColumnsTextVectorizer(
                columns_text=['first_party_type', 'second_party_type', 'facts_of_the_case',  'province_name',],
                columns_multilabel=['heard_by', 'decided_by']

            )),
            ('classifier', self.model)
        ])

        # Fit the classifier
        self.pipeline = self.pipeline.fit(X, Y)

    def predict_proba(self, X):
        res = self.pipeline.predict(X)
        return res
