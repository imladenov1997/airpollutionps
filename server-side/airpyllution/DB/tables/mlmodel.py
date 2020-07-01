from peewee import PrimaryKeyField, CharField, BlobField
from playhouse.postgres_ext import BinaryJSONField, AutoField


def ml_model(Base):
    class MLModel(Base):
        id = AutoField()
        name = CharField(unique=True)
        type = CharField()
        resource = CharField()
        extra_params = BinaryJSONField(null=True)  # used for additional metadata specific to a given type of model
        model_params = BinaryJSONField(null=False)  # used for model parameters and weights serializable to JSON
        extra_data = BlobField(null=True)  # for model parameters/weights serializable to specific file format (e.g. h5)

    MLModel._meta.table_name = 'ml_models'
    MLModel.add_index(MLModel.name, MLModel.type)

    return MLModel
