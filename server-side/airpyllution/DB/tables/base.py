from peewee import Model


def base(db):
    class Base(Model):
        class Meta:
            database = db

    return Base
