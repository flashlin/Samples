import ijson


def enumerate_huge_json_file(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        parser = ijson.parse(file)
        for prefix, event, value in parser:
            if event == 'start_map':
                obj = {}
                for inner_prefix, inner_event, inner_value in parser:
                    if inner_event == 'map_key':
                        key = inner_value
                    elif inner_event in ('string', 'number', 'boolean', 'null'):
                        obj[key] = inner_value
                    elif inner_event == 'end_map':
                        yield obj
                        break
