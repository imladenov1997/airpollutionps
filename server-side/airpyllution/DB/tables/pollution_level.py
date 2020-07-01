from peewee import ForeignKeyField, PrimaryKeyField, Model, FloatField, BooleanField, AutoField


def pollution_level(Base, Dataset=None, Pollutant=None):
    class PollutionLevel(Base):
        id = AutoField()
        dataset_id = ForeignKeyField(Dataset, backref='pollution_levels')
        pollutant_id = ForeignKeyField(Pollutant, backref='pollution_levels')
        pollutant_value = FloatField()
        uncertainty = FloatField(null=True)
        predicted = BooleanField(default=False)

    PollutionLevel._meta.table_name = 'pollution_levels'
    unique_index = PollutionLevel.index(
        PollutionLevel.dataset_id,
        PollutionLevel.pollutant_id,
        unique=True
    )
    PollutionLevel.add_index(unique_index)

    return PollutionLevel
