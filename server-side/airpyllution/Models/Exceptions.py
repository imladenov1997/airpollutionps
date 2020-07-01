class WrongNumberOfFeatures(Exception):
    """
    Exception when there is a wrong number of features for testing or retraining a model
    """
    def __init__(self, message):
        super().__init__(message)