class Helpers:
    """
    Class with a set of helper functions
    """
    @staticmethod
    def are_params_missing(body, required_params):
        return len(list(filter(lambda x: x not in body, required_params))) > 0  # assume no param is missing
