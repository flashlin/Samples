
def dict_to_obj(dictionary, obj_type):
    obj_keys = obj_type.__annotations__.keys()
    obj = obj_type()
    for item_key in dictionary.keys():
        if item_key in obj_keys:
            setattr(obj, item_key, dictionary[item_key])
    return obj

def get_obj_keys(obj: object):
    return vars(obj).keys()

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
