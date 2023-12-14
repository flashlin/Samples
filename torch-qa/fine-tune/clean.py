import os
import re
from collections import defaultdict
from io_utils import query_ext_files


if __name__ == '__main__':
    unique_names = set()
    pattern = r'^(.+?)-(\d+)\.md$'
    name_count = defaultdict(list)

    file_names = query_ext_files('data', ['.md'])
    for file_name in file_names:
        match = re.search(pattern, file_name)
        if match:
            name = match.group(1)
            number = int(match.group(2))
            name_count[name].append(number)

    duplicates = {k: min(v) for k, v in name_count.items() if len(v) > 1}
    print(duplicates)

    for name, number in duplicates.items():
        os.remove(f"{name}-{number}.md")