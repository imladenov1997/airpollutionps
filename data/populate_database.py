import json
import pandas
import random
import numpy as np
import json

from airpyllution import MainTransformer
from airpyllution import Transformers
from airpyllution import DBManager

with open('configTwo.json') as file:
    dataset_one = json.load(file)

with open('configOne.json') as file:
    dataset_two = json.load(file)

data_transformer = MainTransformer(config=dataset_one)
data_transformer.add_transformer(Transformers.WEATHER_TRANSFORMER)
data_transformer.add_transformer(Transformers.POLLUTANT_TRANSFORMER)
data_transformer.transform()
dataset_centre = data_transformer.get_dataset()

data_transformer = MainTransformer(config=dataset_two)
data_transformer.add_transformer(Transformers.WEATHER_TRANSFORMER)
data_transformer.add_transformer(Transformers.POLLUTANT_TRANSFORMER)
data_transformer.transform()
dataset_a33 = data_transformer.get_dataset()

length_centre = dataset_centre.shape[0]
length_a33 = dataset_a33.shape[0]

dataset_centre['Longitude'] = -1.463484 
dataset_centre['Latitude'] = 50.920265
dataset_a33['Longitude'] = -1.395778
dataset_a33['Latitude'] = 50.908140

print(dataset_centre)

DBManager.insert_dataset(dataset_centre, dataset_one)
DBManager.insert_dataset(dataset_a33, dataset_one)