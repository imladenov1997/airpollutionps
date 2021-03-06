import GPy
from .BaseGP import BaseGP


class GaussianProcesses(BaseGP):
    """
    Implementation of Full GP
    """
    VARIANCE = 0.5
    LENGTHSCALE = 1.0
    TYPE = 'Full Gaussian Processes'
    RESOURCE = 'GPy'
    MODEL = GPy.models.GPRegression
    PATH = 'GP/'

    def __init__(self):
        super().__init__()

    def _init_kernel(self, input_dim=None, custom=None):
        """
        Function for initialising the RBF kernel
        :param input_dim: dimensionality of the dataset input
        :param custom: None | GPy.kern instance - custom kernel that is input
        :return: GPy.kern instance
        """
        if custom is None:
            self.kernel = GPy.kern.RBF(input_dim=input_dim, variance=self.VARIANCE,
                                       lengthscale=self.LENGTHSCALE)
        else:
            self.kernel = GPy.kern.Kern.from_dict(custom)

        return self.kernel

    @staticmethod
    def new_from_json(model_params, *args, **kwargs):
        """
        Function for creating a Gaussian Processes model from JSON or dict
        :param model_params: JSON | dict
        :param args: list
        :param kwargs: dict
        :return: (GaussianProcesses, None) | (None, str) - str is an error message
        """
        extra_params = args[0]
        full_gp = GaussianProcesses()
        result, msg = full_gp.load_from_json(model_params, extra_params)

        if result:
            return full_gp, None
        else:
            return None, msg