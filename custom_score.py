from sklearn.metrics import f1_score
from rampwf.score_types.classifier_base import ClassifierBaseScoreType


class F1(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='f1', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='micro')