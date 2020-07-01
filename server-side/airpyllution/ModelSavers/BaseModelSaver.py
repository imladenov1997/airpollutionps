class AbstractModelSaver:
    def __init__(self, config):
        if 'save' in config:
            self.config = config['save']

        
