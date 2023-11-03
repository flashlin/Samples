from dataclasses import dataclass, fields


def dict_to_obj(dictionary: dict, obj_type):
    obj_keys = obj_type.__annotations__.keys()
    obj = obj_type()
    for item_key in dictionary.keys():
        if item_key in obj_keys:
            setattr(obj, item_key, dictionary[item_key])
    return obj


def create_dataclass(dataclass_type, **kwargs):
    data_fields = fields(dataclass_type)
    # print(f"{[field.name for field in data_fields]=}")
    # print(f"{[kwargs.keys()]=}")
    new_entity = dataclass_type(*[kwargs.get(field.name, None) for field in data_fields])
    return new_entity


def get_obj_keys(obj: object):
    return vars(obj).keys()


def obj_to_dict(obj: object):
    dictionary = {}
    for key in get_obj_keys(obj):
        value = getattr(obj, key)
        dictionary[key] = value
    return dictionary


def dict_to_dynamic_object(dictionary: dict):
    return type('DynamicObject', (), dictionary)()


def clone_to_dynamic_object(obj, dictionary: dict):
    dynamic_obj = type('DynamicObject', (), {})()
    for key in get_obj_keys(obj):
        value = getattr(obj, key)
        setattr(dynamic_obj, key, value)
    for key in dictionary.keys():
        setattr(dynamic_obj, key, dictionary[key])
    return dynamic_obj


def dump_obj(obj: object):
    json = "{"
    values = []
    for key in get_obj_keys(obj):
        value = getattr(obj, key)
        values.append(f'"{key}":"{value}"')
    json += ",".join(values)
    json += "}"
    return json


def dump(any_t):
    if isinstance(any_t, list):
        json = "["
        values = []
        for item in any_t:
            values.append(dump(item))
        json += ",".join(values)
        json += "]"
        return json
    return dump_obj(any_t)
