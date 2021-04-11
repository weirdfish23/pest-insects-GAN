import json

def read_config(filename='config.json' ):
    with open(filename) as f:
        data = json.load(f)
    return data