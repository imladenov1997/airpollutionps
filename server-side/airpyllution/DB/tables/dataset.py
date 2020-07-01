from peewee import Model, PrimaryKeyField, DateTimeField, FloatField
from playhouse.postgres_ext import BinaryJSONField, DecimalField, DoubleField, AutoField


def dataset(Base):
    class Dataset(Base):
        id = AutoField()
        datetime = DateTimeField(null=False)
        longitude = DoubleField(null=False)
        latitude = DoubleField(null=False)
        data = BinaryJSONField(default={})

    Dataset._meta.table_name = 'datasets'
    Dataset.add_index(Dataset.longitude, Dataset.latitude)

    return Dataset
