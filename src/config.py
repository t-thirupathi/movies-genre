from pathlib import Path
import json 

def load_config():
    global config
    FILE_DIR = Path(__file__).resolve().parent

    try:
        with open(FILE_DIR / 'config.json') as f:
            config = json.load(f)
    except ValueError:
        sys.exit('Invalid config file')
    
    if 'DATA_DIR' not in config or config['DATA_DIR'] == '':
        config['DATA_DIR'] = FILE_DIR / '../data'
    else:
        config['DATA_DIR'] = Path(config['DATA_DIR']).resolve()

    if 'MODELS_DIR' not in config or config['MODELS_DIR'] == '':
        config['MODELS_DIR'] = FILE_DIR / '../models'
    else:
        config['MODELS_DIR'] = Path(config['MODELS_DIR']).resolve()

    assert config['DATA_DIR'].exists()
    assert config['MODELS_DIR'].exists()

    return config


