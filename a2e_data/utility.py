import argparse
from datetime import datetime, timezone


def timestamp_to_date_time(d):
    return datetime.fromtimestamp(float(d), tz=timezone.utc)


def get_recursive_config(config: dict, *args, **kwargs):
    for arg in args:
        if arg in config:
            config = config[arg]
        else:
            if 'default' not in kwargs:
                raise ValueError('Config with name "' + str(arg) + '" is not set.')
            else:
                return kwargs['default']

    return config


def str2bool(value: str) -> bool:
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Unexpected string "{value}" that cannot be converted to a boolean.')
