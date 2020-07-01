import json
import sys
from functools import reduce
import os

def generate_env_files(paths):
    env_files = []
    envs = list(map(lambda path: generate_env(path), paths))
    env_vars = list(reduce(lambda envs, env: envs + env, envs, []))

    with open('envs.env', 'w+') as file:
        for env_var in env_vars:
            file.write(env_var + os.linesep)


def generate_env(path):
    envs = []
    with open(path) as db_config_file:
        db_config = json.load(db_config_file)
        envs = []
        
        for key in db_config:
            envs.append(key + '=' + db_config[key])
    
    return envs

generate_env_files(sys.argv[1:])