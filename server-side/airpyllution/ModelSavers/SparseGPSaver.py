import GPy
import numpy as np

from .BaseModelSaver import AbstractModelSaver


class SparseGPSaver(AbstractModelSaver):
    @staticmethod
    def save_model(filename, model):
        """"
            This method assumes that a GPy model was exported
        """
        if isinstance(model, GPy.models.GPRegression):
            np.save('../SavedModels/' + filename, model.param_array)
            return True

        return False