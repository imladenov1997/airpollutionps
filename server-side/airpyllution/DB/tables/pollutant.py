from peewee import Model, PrimaryKeyField, DateTimeField, FloatField, CharField, AutoField


def pollutant(Base):
    class Pollutant(Base):
        id = AutoField()
        name = CharField(unique=True, index=True)

    Pollutant._meta.table_name = 'pollutants'

    return Pollutant
