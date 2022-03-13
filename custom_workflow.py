import os

from rampwf.utils.importing import import_module_from_source
from rampwf.workflows.classifier import Classifier as RCLF

class Classifier(RCLF):

    def train_submission(self, module_path, X_array, y_array, train_is=None,
                         prev_trained_model=None):
        if train_is is None:
            train_is = slice(None, None, None)
        classifier = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        clf = classifier.Classifier()
        if prev_trained_model is None:
            clf.fit(X_array.iloc[train_is], y_array[train_is])
        else:
            clf.fit(
                X_array.iloc[train_is], y_array[train_is], prev_trained_model)
        return clf