import json
import os
from celery import Celery
from flask import Flask, request
from flask_cors import CORS
from api.ModelAPI import ModelApi
from api.DatasetsAPI import DatasetsApi
from utils.Errors import Errors
from utils.Response import response_success, response_failure
from utils.Helpers import Helpers
import airpyllution

# Load database
DBManager = airpyllution.DBManager

# Initialise Flask app and add CORS
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Add message broker
web_app_config = {
    'broker_url': os.environ['BROKER_URL'],
    'result_backend': os.environ['RESULT_BACKEND']
}

if web_app_config is None:
    app.config.update(
        broker_url='amqp://127.0.0.1:5672',
        result_backend='amqp://127.0.0.1:5672'
    )
else:
    app.config.update(
        broker_url=web_app_config['broker_url'],
        result_backend=web_app_config['result_backend']
    )


celery = Celery(
    app.name,
    backend=app.config['result_backend'],
    broker=app.config['broker_url']
)
celery.conf.update(app.config)


class ContextTask(celery.Task):
    def __call__(self, *args, **kwargs):
        with app.app_context():
            return self.run(*args, **kwargs)


celery.Task = ContextTask


class ConcurrentTaskApi:
    """
    Class for all API endpoinds that will execute as background tasks
    """
    @staticmethod
    @celery.task
    def concurrent_create_model(model_name, body):
        return ModelApi.create_model(model_name, body)

    @staticmethod
    @celery.task
    def concurrent_make_predictions(body):
        return ModelApi.make_predictions(body)

    @staticmethod
    @celery.task
    def concurrent_train_model(model_name, body):
        return ModelApi.train_model(model_name, body)


@app.route('/api', methods=['GET'])
def home():
    """
    Default home function
    :return: JSON
    """
    return response_success('Home')


@app.route('/api/create-model/<name>', methods=['POST'])
def create_model(name):
    """
    Function to create a model from given type and train it on a dataset with given parameters
    :param name: unique name of the created model
    :return: JSON
    """

    REQUIRED_PARAMS = [
        'type',
        'range',
        'locations',
        'pollutant'
    ]

    body = request.json
    missing_param = Helpers.are_params_missing(body, REQUIRED_PARAMS)  # assume no param is missing

    if missing_param:
        return response_failure(Errors.MISSING_PARAM.value)

    if not isinstance(body['range'], dict):
        return response_failure(Errors.WRONG_PARAM.value)

    result = ConcurrentTaskApi.concurrent_create_model.delay(name, body)

    return response_success(True)


@app.route('/api/get-pollution', methods=['POST'])
def get_pollution_levels():
    """
    Function for getting the pollution levels in a given time range for given locations
    This method is POST due to the high number of parameters included
    :return: JSON
    """
    REQUIRED_PARAMS = [
        'range',
        'locations',
        'pollutant'
    ]

    body = request.json
    missing_param = Helpers.are_params_missing(body, REQUIRED_PARAMS)  # assume no param is missing

    if missing_param:
        return response_failure(Errors.MISSING_PARAM.value)

    if not isinstance(body['range'], dict):
        return response_failure(Errors.WRONG_PARAM.value)

    dataset = DatasetsApi.get_dataset(body, use_dataframe=False)

    return response_success(dataset)


@app.route('/api/predict', methods=['POST'])
def make_predictions():
    """
    Function for making predictions based on the datetime and locations given for a single pollutant, type of model
    is also required (CNN/FullGP/SparseGP)
    :return: JSON
    """
    REQUIRED_PARAMS = [
        'name',
        'range',
        'locations',
        'pollutant'
    ]

    body = request.json
    missing_param = Helpers.are_params_missing(body, REQUIRED_PARAMS)  # assume no param is missing

    if missing_param:
        return response_failure(Errors.MISSING_PARAM.value)

    if not isinstance(body['range'], dict):
        return response_failure(Errors.WRONG_PARAM.value)

    result = ConcurrentTaskApi.concurrent_make_predictions.delay(body)

    return response_success(True)


@app.route('/api/train/<model_name>', methods=['POST'])
def train_model(model_name):
    """
    Route for training a model
    :param model_name: str - model's name
    :return: JSON
    """
    REQUIRED_PARAMS = [
        'range',
        'locations',
        'pollutant'
    ]

    body = request.json
    are_params_missing = Helpers.are_params_missing(body, REQUIRED_PARAMS)

    if are_params_missing:
        return response_failure(Errors.MISSING_PARAM.value)

    if not isinstance(body['range'], dict):
        return response_failure(Errors.WRONG_PARAM.value)

    if not (isinstance(body['locations'], dict) or isinstance(body['locations'], list)):
        return response_failure(Errors.WRONG_PARAM.value)

    ConcurrentTaskApi.concurrent_train_model.delay(model_name, body)

    return response_success(True)


@app.route('/api/get-model-params/<name>', methods=['GET'])
def get_model_params(name):
    """
    Route for getting model parameters of a given model
    :param name: str - name of the model
    :return: JSON
    """
    model = ModelApi.get_model_params(name)
    if model is not None:
        return response_success(model)

    return response_failure(Errors.NO_SUCH_MODEL.value)


@app.route('/api/insert-measurement', methods=['POST'])
def insert_single_measurement():
    """
    Route for adding a single measurement, this endpoint could be used by external sensors, devices, etc. for adding
    to the database
    :return: JSON
    """
    REQUIRED_PARAMS = [
        'date_time',
        'longitude',
        'latitude',
        'pollutant',
        'pollution_value'
    ]

    body = request.json
    missing_param = Helpers.are_params_missing(body, REQUIRED_PARAMS)  # assume no param is missing

    if missing_param:
        return response_failure(Errors.MISSING_PARAM.value)

    is_successful, err = DatasetsApi.insert_single_measurement(body)

    # Assume instance has not been inserted
    response_msg = {
        'instance_inserted': err
    }

    if is_successful:
        response_msg = {
            'instance_inserted': True
        }
        return response_success(response_msg)

    return response_failure(response_msg)


@app.route('/api/insert-dataset', methods=['POST'])
def insert_dataset():
    """
    Route for adding a whole dataset to the database, not preferable to use it with large datasets, dataset must be in
    CSV
    :return: JSON
    """
    pollutant = None
    weather = None
    dataset_metadata = None
    try:
        pollutant = request.files['pollutant']
        weather = request.files['weather']
        dataset_metadata = request.files['metadata']
    except:
        print('no such file')

    if pollutant is None:
        return response_failure({
            'file': 'pollutant'
        })

    if weather is None:
        return response_failure({
            'file': 'weather'
        })

    if dataset_metadata is None:
        return response_failure({
            'file': 'metadata'
        })

    files = {
        'pollutant': pollutant,
        'weather': weather,
        'metadata': dataset_metadata
    }

    result, err = DatasetsApi.insert_dataset(files)

    if err:
        return response_failure(err)

    return response_success('inserted')


@app.route('/api/get-models/<model_type>', methods=['GET'])
def get_models(model_type):
    """
    Route for getting models that are of given model type (CNN, GP, SparseGP up to date)
    :param model_type:
    :return: JSON
    """
    models, err = ModelApi.get_models_by_type(model_type)
    if models is not None:
        return response_success(models)

    return response_failure(err)


@app.route('/api/get-coordinates', methods=['GET'])
def get_coordinates():
    """
    Route for getting all coordinate pairs existing in the DB
    :return: JSON
    """
    return response_success(DatasetsApi.get_coordinates())


@app.route('/api/get-pollutants', methods=['GET'])
def get_pollutants():
    """
    Route for getting all pollutants existing in the DB
    :return: JSON
    """
    return response_success(DatasetsApi.get_pollutants())


@app.route('/api/get-models', methods=['GET'])
def get_all_models():
    """
    Route for getting all models' names and types only from the DB
    :return: JSON
    """
    models = ModelApi.get_all_models()

    if models is None:
        return response_failure(models)

    return response_success(models)


@app.route('/api/predict-single-instance', methods=['POST'])
def predict_single_instance():
    """
    Route for creating a new arbitrary instance and making prediction on its
    :return: JSON
    """
    REQUIRED_PARAMS = [
        'name',
        'date_time',
        'longitude',
        'latitude',
        'pollutant'
    ]

    body = request.json
    missing_param = Helpers.are_params_missing(body, REQUIRED_PARAMS)

    if missing_param:
        return response_failure(Errors.MISSING_PARAM.value)

    is_successful, predictions = ModelApi.make_single_prediction(body)

    if not is_successful:
        # TODO update error message
        return response_failure('Something went wrong')

    return response_success(predictions)


@app.route('/api/get-model-types', methods=['GET'])
def get_model_types():
    """
    Route for getting all supported models' types, currently only CNN, FullGP, SparseGP
    :return: JSON
    """
    model_types = ModelApi.get_model_types()

    if not isinstance(model_types, list):
        return response_failure(Errors.NO_DATA.value)

    return response_success(model_types)


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
