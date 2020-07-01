import json

import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential, Input, Model, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import Graph
from tensorflow import Session

from airpyllution.Models.Exceptions import WrongNumberOfFeatures
from ..Metrics.Metrics import Metrics
from .BaseModel import AbstractBaseModel
from ..Utils.Errors import Errors
from ..Utils.ImageEncoder import ImageEncoder


class ConvolutionalNeuralNetwork(AbstractBaseModel):
    TYPE = 'Convolutional Neural Network'
    RESOURCE = 'Keras'

    def __init__(self, seq_length=None):
        """"
        Constructor
        :param seq_length - height of each image or time period of measurements (e.g. 24)
        :return None
        """
        self.batch_size = 40
        self.epochs = 100
        self.predict_iterations = 25  # how many predictions with dropout, used for uncertainty calculation
        self.seq_length = seq_length if seq_length is not None else 24
        self.n_features = None
        self.is_built = False
        self.stats = {
            'n_instances_trained': 0,
            'dataset_stats': {}
        }
        self.graph = Graph()
        with self.graph.as_default():
            self.sess = Session()

    def train(self, X_train, y_train, stats=None):
        """"
        Training method, fits the CNN
        :param stats:
        :param X_train - a DataFrame with features
        :param y_train - a DataFrame with measurements
        :return None
        """
        self.n_features = X_train.shape[1]
        self.__build_func(self.seq_length, self.n_features)  # build model
        self.is_built = True

        if self.stats['n_instances_trained'] == 0:
            self.stats['dataset_stats'] = stats
            self.stats['n_instances_trained'] = X_train.shape[0]
        elif self.__check_features(X_train):
            self.update_stats(stats, X_train.shape[0])  # get new stats and number of instances
        else:
            raise WrongNumberOfFeatures('Training set does not have the same set of features model has been trained')

        # Firstly, rescale dataset to support 0-255 range format for images
        # This step is rather here than in normal Data Transformers as it is CNN-specific
        # X_new_train = ImageEncoder.rescale_dataset(X_train, False)
        if X_train.shape[0] <= self.seq_length:
            raise NotEnoughInstancesError('Dataset is insufficient for training a CNN')

        X_train_images = ImageEncoder.generate_image_set(X_train, self.seq_length).reshape(-1, self.seq_length,
                                                                                           self.n_features, 1)
        # ignore first seq_length isntances from test_set as they are in first image
        y_train_set = y_train.iloc[self.seq_length:]

        train_X, valid_X, train_label, valid_label = train_test_split(X_train_images, y_train_set, test_size=0.2,
                                                                      random_state=12)
        with self.graph.as_default():
            with self.sess.as_default():
                self.model.fit(train_X, train_label, batch_size=self.batch_size, epochs=self.epochs,
                               validation_data=(valid_X, valid_label))

    def predict(self, X_test, uncertainty=True):
        """
        Function for making predictions on given dataset
        :param X_test: DataFrame - dataset to make predictions on pollution levels for
        :param uncertainty: bool - whether to output uncertainty along with predictions
        :return: list of floats or tuples of floats
        """
        if not self.__check_features(X_test):
            raise WrongNumberOfFeatures('Set does not have the same set of features model has been trained')

        return self.predict_with_uncertainty(X_test) if uncertainty else self.predict_without_uncertainty(X_test)

    def predict_without_uncertainty(self, X_test):
        """"
        Method for making predictions given a prediction set without measurements
        :param X_text - DataFrame with features
        :return numpy array with prediction for each instance
        """

        if X_test.shape[0] < self.seq_length:
            return [(None, None) for _ in range(X_test.shape[0])]

        # X_new_train = ImageEncoder.rescale_dataset(X_test, False)
        X_test_images = ImageEncoder.generate_image_set(X_test, self.seq_length).reshape(-1, self.seq_length,
                                                                                         self.n_features, 1)

        with self.graph.as_default():
            with self.sess.as_default():
                predictions = [(None, None) for _ in range(self.seq_length)]
                predicted = self.model.predict(X_test_images)

                for x in predicted:
                    predictions.append(x)

                return predictions

    def eval(self, X_test, y_test, error_func=None):
        """"
        Evaluation method to measure error the model produces
        :param X_test - DataFrame with features
        :param y_test - actual measurements
        :param error_func - callback function (should be from Metrics)
        :return result, predictions, y_test_set:
                result - float value giving the error
                predictions - numpy array with actual predictions
                y_test_set - it is necessary to return the test_set used because the first seq_length values are used for first instance
        """

        if not self.__check_features(X_test):
            raise WrongNumberOfFeatures('Set does not have the same set of features model has been trained')

        if X_test.shape[0] < self.seq_length:
            return None, [(None, None) for _ in range(X_test.shape[0])], y_test[['Pollutant']].iloc[:self.seq_length]

        predictions_with_uncertainty = self.predict(X_test)
        predictions = list(map(lambda x: x[0], predictions_with_uncertainty))

        # get two arrays for evaluation
        y_test_set = y_test[['Pollutant']].iloc[self.seq_length:]
        evaluated_predictions = predictions[self.seq_length:]
        if error_func is None or not callable(error_func):
            error_func = Metrics.rmse  # default metrics

        result = error_func(y_test_set, evaluated_predictions)

        return result, predictions, y_test

    @staticmethod
    # The implementation of the following function was used from here
    # https://stackoverflow.com/questions/43529931/how-to-calculate-prediction-uncertainty-using-keras
    def prob_func(model):
        """"
        Probability function for predicting with Uncertainty
        :param model: keras.models.Model - should be a trained one
        :return Backend function (function type depends on the backend of keras, e.g. tensorflow, theano, etc.)

        """
        return K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    # https://stackoverflow.com/questions/43529931/how-to-calculate-prediction-uncertainty-using-keras
    def predict_with_uncertainty(self, X):
        """"
        :param X: DataFrame - dataset whose values should be predicted
        :return predictions: list of tuples (prediction, uncertainty)
                                prediction - predicted value for each instance
                                uncertainty - uncertainty in measurements
        """

        # Check if the dataset is at least self.seq_length - then for one prediction one image with 24 previous values
        # is generated, otherwise just return no predictions
        if X.shape[0] < self.seq_length:
            return [(None, None) for _ in range(X.shape[0])]

        X_test_images = ImageEncoder.generate_image_set(X, self.seq_length).reshape(-1, self.seq_length,
                                                                                    self.n_features, 1)

        with self.graph.as_default():
            with self.sess.as_default():
                f = ConvolutionalNeuralNetwork.prob_func(self.model)  # use prediction function
                # start with first predictions that were used for first image (25th pollution )
                predictions = [(None, None) for _ in range(self.seq_length)]
                for x in X_test_images:
                    result = []

                    for i in range(self.predict_iterations):
                        prediction = f([np.array(x).reshape(-1, self.seq_length, self.n_features, 1), 1])
                        result.append(prediction)
                    result = np.array(result)

                    # Bayesian approximation for prediction and uncertainty
                    prediction = result.mean()
                    prediction = prediction if prediction >= 0 else 0
                    uncertainty = result.std()

                    predictions.append((prediction, uncertainty))

        return predictions

    # Using Keras Sequential Model API
    # Does not predict uncertainty
    def __build(self, seq_length, n_features):
        """"
        Private function for building a sequential model
        That model is easily built, but is simpler and not that flexible
        That's the reason for missing the uncertainty feature
        Architecture is the same with __build_prob() but without the Dropout function at the final layer

        :param seq_length: int - how many instances from X shall form a 2D image (where it determines the height)
        :param n_features: int - how many features one instance hold
        :return None
        """
        # Check if model's architecture has to be built now or is loaded from file
        with self.graph.as_default():
            with self.sess.as_default():
                if not self.is_built:
                    self.model = Sequential()
                    self.model.add(
                        Conv2D(16, kernel_size=(5, 1), strides=1, activation='linear',
                               input_shape=(seq_length, n_features, 1),
                               padding='same'))
                    # self.model.add(BatchNormalization())
                    self.model.add(ELU(alpha=0.1))
                    self.model.add(MaxPooling2D((5, 1), padding='same'))

                    self.model.add(Conv2D(32, kernel_size=(9, 1), strides=1, activation='linear', padding='same'))
                    # self.model.add(BatchNormalization())
                    self.model.add(ELU(alpha=0.1))
                    self.model.add(MaxPooling2D((7, 1), padding='same'))

                    self.model.add(Conv2D(64, kernel_size=(13, 1), strides=1, activation='linear', padding='same'))
                    # self.model.add(BatchNormalization())
                    self.model.add(ELU(alpha=0.1))
                    self.model.add(MaxPooling2D((9, 1), padding='same'))

                    self.model.add(Flatten())
                    self.model.add(Dense(1))

                    self.is_built = True

                self.model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam())

    # Using Keras Functional Model API
    # Predicts Uncertainty with Bayesian Approximation
    def __build_func(self, seq_length, n_features):
        """"
        Private function for building a functional model
        That model is built a bit more different and can be more powerful and flexible compared to sequential model
        Implements uncertainty feature unlike the sequential model
        Architecture is the same with __build() but with the Dropout function at the final layer

        :param seq_length: int - how many instances from X shall form a 2D image (where it determines the height)
        :param n_features: int - how many features one instance hold
        :return None
        """
        # Check if model's architecture has to be built now or is loaded from file
        with self.graph.as_default():
            with self.sess.as_default():
                if not self.is_built:
                    # Input layer
                    input = Input(shape=(seq_length, n_features, 1))

                    # layer 1
                    conv_template_1 = Conv2D(16, kernel_size=(5, 1), strides=1, activation='linear', padding='same')(
                        input)
                    elu_1 = ELU(alpha=0.1)(conv_template_1)
                    max_pool_1 = MaxPooling2D((5, 1), padding='same')(elu_1)

                    # layer 2
                    conv_template_2 = Conv2D(32, kernel_size=(9, 1), strides=1, activation='linear', padding='same')(
                        max_pool_1)
                    elu_2 = ELU(alpha=0.1)(conv_template_2)
                    max_pool_2 = MaxPooling2D((7, 1), padding='same')(elu_2)

                    # layer 3
                    conv_template_3 = Conv2D(64, kernel_size=(13, 1), strides=1, activation='linear', padding='same')(
                        max_pool_2)
                    elu_3 = ELU(alpha=0.1)(conv_template_3)
                    max_pool_3 = MaxPooling2D((9, 1), padding='same')(elu_3)

                    # Flatten output
                    flatten = Flatten()(max_pool_3)
                    dense = Dense(2)(flatten)
                    dropout = Dropout(0.5)(dense, training=True)  # Apply dropout technique for uncertainty
                    output = Dense(1)(dropout)
                    self.model = Model(input, output)

                    self.is_built = True

                self.model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam())

    def save_model(self, config):
        """"
        Saves model's architecture and weights so that it can be reused
        Uses 'persistence' field in config.json

        Models are saved in SavedModels/CNN/ directory

        Does not save datasets within the model
        :param config - dict taken from ConfigReader.CONFIG
        :return tuple (is_successful, error_message)
                is_successful: boolean - whether the saving is successful or not
                error_message: str|None - None when successful, otherwise shows what caused the error
        """
        if not isinstance(self.model, Model):
            return False, Errors.WRONG_INSTANCE.value

        if not isinstance(config, dict):
            return False, Errors.WRONG_CONFIG.value

        save_data = None

        # Check where to store model
        if 'persistence' in config:
            save_data = config['persistence']
        else:
            return False, Errors.MODEL_NO_NAME.value

        # Further validation on necessary data to save model
        if 'modelName' in save_data and isinstance(save_data['modelName'], str):
            dir = './airpyllution/SavedModels/CNN/'
            # Save architecture
            path_json = dir + save_data['modelName'] + '.json'

            with self.graph.as_default():
                with self.sess.as_default():
                    # Save model and some metadata
                    model = {
                        'type': ConvolutionalNeuralNetwork.TYPE,
                        'resource': ConvolutionalNeuralNetwork.RESOURCE,
                        'params': json.loads(self.model.to_json()),
                        'features': self.n_features,
                        'stats': self.stats
                        # it's better to be outside params as this is not part of NN's architecture
                    }

                    with open(path_json, 'w+') as json_obj:
                        try:
                            json.dump(model, json_obj, sort_keys=False, indent=4)
                        except:
                            return False, Errors.FILE_NOT_SAVED.value

                    # Save weights in a separate file as Keras requires it
                    path_weights = dir + save_data['modelName'] + '.h5'

                    try:
                        self.model.save_weights(path_weights)
                    except:
                        return False, Errors.MODEL_DATA_NOT_SAVED.value

            return True, None

        return False, Errors.NO_MODEL_DATA.value

    def load_model(self, config):
        """"
        Loads model's architecture and weights from existing files, no training required, but strill necessary to build
        the model (more specifically compile it)
        Uses 'loadedModel' field in config.json
        Uses different field in config.json as in one operation a model can be loaded and then further saved
        Once a model is loaded, it can be used as normal

        Models are loaded from SavedModels/CNN/ directory and requires .json and .h5 files for each model, each with
        the same name

        Does not save datasets within the model
        :param config - dict taken from ConfigReader.CONFIG
        :return tuple (is_successful, error_message)
                is_successful: boolean - whether the loading is successful or not
                error_message: str|None - None when successful, otherwise shows what caused the error
        """
        if not isinstance(config, dict):
            return False, Errors.WRONG_CONFIG.value

        loading_data = None

        if 'loadedModel' in config:
            loading_data = config['loadedModel']
        else:
            return False, Errors.NO_SUCH_MODEL.value

        if 'modelName' in loading_data:
            architecture_path = './airpyllution/SavedModels/CNN/' + loading_data['modelName'] + '.json'
            weights_path = './airpyllution/SavedModels/CNN/' + loading_data['modelName'] + '.h5'

            with open(architecture_path, 'r') as file:
                try:
                    json_obj = json.load(file)
                except:
                    return False, Errors.FILE_NOT_LOADED.value

                if 'params' in json_obj:
                    with self.graph.as_default():
                        with self.sess.as_default():
                            self.model = model_from_json(json.dumps(json_obj['params']))
                            self.model.load_weights(weights_path)
                            self.is_built = True
                else:
                    return False, Errors.NO_MODEL_PARAMS.value

                if 'features' in json_obj:
                    self.n_features = json_obj['features']
                else:
                    return False, Errors.NO_MODEL_DATA.value

                if 'stats' in json_obj:
                    self.stats = json_obj['stats']

                return True, None

        return False, Errors.NO_SUCH_MODEL.value

    def model_to_json(self):
        """"
        Function that returns architecture, weights and additional metadata for building the trained net

        :return model_params, extra_params - both are JSON strings ready to be exported as a .json document or added
        to a database
        """
        model_params = {}
        extra_params = {}
        if self.model is not None:
            with self.graph.as_default():
                with self.sess.as_default():
                    architecture = self.model.to_json()
                    weights = [ls.tolist() for ls in self.model.get_weights()]

                    model_params = {
                        'architecture': architecture,
                        'weights': weights
                    }

                    extra_params = {
                        'sequence_length': self.seq_length,
                        'n_features': self.n_features,
                        'stats': self.stats
                    }

        model_params = json.dumps(model_params)
        extra_params = json.dumps(extra_params)

        return model_params, extra_params

    def load_from_json(self, json_model_data):
        """"
        Function that creates a trained neural net from the architecture and model provided
        This function assumes that metadata (sequence_length and n_features) is already set

        :param json_model_data dict or stringified JSON having two fields - architecture and weights
        :return boolean - whether a model is successfully loaded

        :parm json_model_data
        """
        model_params = json_model_data if isinstance(json_model_data, dict) else json.loads(json_model_data)
        if 'architecture' in model_params and 'weights' in model_params:
            with self.graph.as_default():
                with self.sess.as_default():
                    self.model = model_from_json(model_params['architecture'])
                    self.model.set_weights(model_params['weights'])
                    self.is_built = True

            return True

        return False

    @staticmethod
    def new_from_json(json_model_params, json_extra_params):
        """"
        Static method implementing factory pattern for creating a new CNN with given model parameters and metadata

        :param json_model_params - dict or stringified JSON having two fields - architecture and weights
        :param json_extra_params - dict or stringified JSON having two fields - sequence_length and n_features, this is
        the metadata
        :return (ConvolutionalNeuralNetwork, None) | (None, str)
        First value of the tuple is the network if successful, second is the error message if any
        """
        extra_params = json_extra_params if isinstance(json_extra_params, dict) else json.loads(json_extra_params)
        if 'sequence_length' not in extra_params:
            return None, Errors.MISSING_PARAM.value

        cnn = ConvolutionalNeuralNetwork(extra_params['sequence_length'])
        cnn.n_features = extra_params['n_features']

        if 'stats' in extra_params:
            cnn.stats = extra_params['stats']

        cnn.load_from_json(json_model_params)
        return cnn, None

    def update_stats(self, new_stats, n_new_instances):
        """
        Method for updating mean value of each feature in the dataset,
        it is important for predicting new datasets that have 0 mean by default
        :param new_stats: dict
        :param n_new_instances:
        :return:
        """
        n_cur_instances = self.stats['n_instances_trained']
        total = n_cur_instances + n_new_instances

        # Get weighted average of both
        weight_current_instances = n_cur_instances / total
        weight_new_instances = n_new_instances / total

        updated_dataset_stats = {}

        if 'dataset_stats' in self.stats and isinstance(self.stats['dataset_stats'], dict):
            for key, value in self.stats['dataset_stats'].items():
                if key in new_stats and 'mean' in new_stats[key] and 'std' in new_stats[key]:
                    updated_dataset_stats[key] = {}
                    updated_dataset_stats[key]['mean'] = new_stats[key]['mean'] * weight_new_instances
                    updated_dataset_stats[key]['std'] = new_stats[key]['std'] * weight_new_instances
                else:
                    print(key)
                    continue

                if 'mean' in value and 'std' in value:
                    updated_dataset_stats[key]['mean'] += value['mean'] * weight_current_instances
                    updated_dataset_stats[key]['std'] += value['std'] * weight_current_instances

        self.stats['n_instances_trained'] = total
        self.stats['dataset_stats'] = updated_dataset_stats

    def __check_features(self, new_dataset):
        """
        Compare features of the dataset model has been trained on and given dataset's features
        :param new_dataset: DataFrame
        :return: bool
        """
        if isinstance(self.stats, dict) and len(self.stats['dataset_stats']) != len(new_dataset.columns):
            return False

        for key in self.stats['dataset_stats'].keys():
            if key not in new_dataset:
                return False
        else:
            return True


class NotEnoughInstancesError(Exception):
    """
    Error when input does not have enough instances to generate even a single image
    """
    def __init__(self, message):
        super().__init__(message)
