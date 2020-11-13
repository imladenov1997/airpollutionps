from enum import Enum


class Errors(Enum):
    WRONG_INSTANCE = 'Error: Wrong instance is given'
    WRONG_CONFIG = 'Error: Wrong config given'
    MODEL_NO_NAME = 'Error: modelName parameter not given'
    SAVE_ERROR = 'Error when saving model'
    NO_SUCH_MODEL = 'Error when loading model'
    NO_VALID_KERNEL = 'Error: Kernel not valid'
    NO_KERNEL = 'Error: No Kernel available'
    NO_MODEL_DATA = 'Error: Model data not found'
    NO_MODEL_PARAMS = 'Error: Model parameters not found'
    NO_DATASETS_AVAILABLE = 'Error: No datasets available'
    FILE_NOT_LOADED = 'Error: File not loaded'
    FILE_NOT_SAVED = 'Error: File not saved'
    MODEL_DATA_NOT_SAVED = 'Error: Model data not saved'
    JSON_NOT_LOADED = 'Error: JSON failed to load'
    DB_CONN_FAIL = 'Error: Could not connect to the database'
    MISSING_PARAM = 'Error: Missing parameter'
    NO_DATETIME = 'Error: No datetime provided'
    NO_LOCATION = 'Error: Location coordinates missing'
    PREDICTED_NOT_BOOL = 'Error: predicted parameter should be Boolean'
    NO_POLLUTANT = 'Warning: No pollutant is given, but instance is inserted without a pollution value'
    MISSING_CONFIG_DATA = 'Error: Required data missing in config'
    DATASET_RECORD_RETRIEVAL_FAILURE = 'Error: Instance retrieval failed'
    POLLUTION_LEVEL_INSERTION_FAILED = 'Error: Pollution level insertion failed...'
    POLLUTANT_INSERTION_OR_RETRIEVAL_FAILED = 'Error: Pollution type insertion/retrieval failed...'
    NO_POLLUTION_LEVEL = 'Warning: No pollution inserted'