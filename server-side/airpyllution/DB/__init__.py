from .DBManager import DBManager

print('Connecting...')
DBManager.connect()
DBManager.create_tables()
