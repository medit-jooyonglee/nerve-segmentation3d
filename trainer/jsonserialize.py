import json
import base64
import numpy as np


class_name_key = 'class_name'

def update_dict_json_format(original_dict, target_dict):

    res = object2json(original_dict)
    if res is not None:
        return res

    for key, value in original_dict.items():
        if isinstance(value, dict):
            if not key in target_dict:
                target_dict[key] = dict()
            update_dict_json_format(value, target_dict[key])
        else:
            # if isinstance(value, np.ndarray):
            #     target_dict[key] = to_json_ndarray(value)
            if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, str):
                target_dict[key] = value
            elif isinstance(value, (int, float, bool, str)):
                target_dict[key] = value
            elif isinstance(value, object):
                if hasattr(value, "__dict__"):
                    json_dict = dict()
                    update_dict_json_format(json_dict, value.__dict__)
                    target_dict[key] = json_dict
                else:
                    target_dict[key] = object2json(value)

                    # raise ValueError("not supported value : {}".format(type(value)))
            # else:?
            elif value is None:
                target_dict[key] = "None"
            else:
                target_dict[key] = object2json(value)
                # raise ValueError(value)


def restore_dict_json_format(original_dict, target_dict):
    # ob_to_json()
    res = json2object(original_dict)
    if res is not None:
        target_dict.update(res)
        return res

    for key, value in original_dict.items():
        if isinstance(value, dict):
            if not key in target_dict:
                target_dict[key] = dict()
        # if isinstance(value, dict):
            # in case converted jon frmo objecg(
            if class_name_key in value:
                target_dict[key] = json2object(value)
            else:
                # general dict
                restore_dict_json_format(value, target_dict[key])
        else:
            if isinstance(value, np.ndarray):
                target_dict[key] = from_json_ndarray(value)
            if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, str):
                target_dict[key] = value
            elif isinstance(value, (int, float, bool, str)):
                target_dict[key] = value
            elif isinstance(value, object):
                if hasattr(value, "__dict__"):
                    json_dict = dict()
                    restore_dict_json_format(json_dict, value.__dict__)
                    target_dict[key] = json_dict
                else:
                    raise ValueError("not supported value : {}".format(type(value)))
            elif value is None:
                target_dict[key] = "None"
            else:
                raise ValueError(value)


def to_json(inst:dict):
    json_readable = dict()
    update_dict_json_format(inst, json_readable)
    return json_readable


def from_json(json_redable:dict):
    json_readable = dict()
    restore_dict_json_format(json_redable, json_readable)
    return json_readable


def object2json(obj):
    if isinstance(obj, np.ndarray):
        return to_json_ndarray(obj)
    else:
        # not implemented
        return None


def json2object(json_dict):
    clas_name = json_dict.get(class_name_key, '')

    if clas_name == np.ndarray.__name__:
        return from_json_ndarray(json_dict)
    else:
        return None


def to_json_ndarray(numpy_array:np.ndarray):
    return {
        'data':  base64.b64encode(numpy_array.tobytes()).decode(),
        'shape': numpy_array.shape,
        'dtype': str(numpy_array.dtype),
        class_name_key: numpy_array.__class__.__name__
    }


def from_json_ndarray(json_data):
    if 'data' in json_data and \
            'shape' in json_data and \
            'dtype' in json_data and \
            class_name_key in json_data:
        # json_data['class'] == 'nd_array'
        arr = np.frombuffer(base64.b64decode(json_data['data']), dtype=json_data['dtype'])
        return arr.reshape(json_data['shape'])
    else:
        return np.array([])

