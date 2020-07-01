from enum import Enum


class Errors(Enum):
    MISSING_PARAM = "Missing parameter"
    WRONG_PARAM = 'Wrong parameter'
    NO_SUCH_MODEL = 'Error: No such model'
    NO_SUCH_MODEL_TYPE = 'Error: Given model type does not exist'
    NO_SUCH_FILE = 'Error: No such file'
    MISSING_METADATA = 'Error: Metadata is missing from uploaded file'
    WRONG_INSTANCE = 'Error: Wrong instance given'
    NO_DATA = 'Error: No data'
    MISSING_BODY = 'Error: No body given'
    MISSING_DATETIME = 'Error: No datetime string given'
    WRONG_LONGITUDE = 'Error: Wrong longitude param given'
    WRONG_LATITUDE = 'Error: Wrong latitude param given'
    NO_RANGE = 'Error: No date and time range given'
    NO_LOCATIONS = 'Error: No locations given in the right format'
    NO_MODEL_TYPE_GIVEN = 'Error: No model type given'
