import os
from datetime import datetime
from functools import wraps
from http import HTTPStatus
import json
import shutil


def create_dirs(dirpath):
    """Creating directories."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_json(filepath):
    """Load a json file."""
    with open(filepath, "r") as fp:
        json_obj = json.load(fp)
    return json_obj


def save_dict(d, filepath):
    """Save dict to a json file."""
    with open(filepath, 'w') as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""
    @wraps(f)
    def wrap(*args, **kwargs):
        results = f(*args, **kwargs)

        # Construct response
        response = {
            'message': results['message'],
            'method': request.method,
            'status-code': results['status-code'],
            'timestamp': datetime.now().isoformat(),
            'url': request.url,
        }

        # Add data
        if results['status-code'] == HTTPStatus.OK:
            response['data'] = results['data']

        return response

    return wrap
