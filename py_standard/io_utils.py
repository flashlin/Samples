import os
import glob


def query_sub_files(folder: str, ext_list: list[str]):
    """
    :param folder:
    :param ext_list: ['.png', '.jpg']
    :return:
    """
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(ext_list):
                file_path = os.path.join(root, file)
                yield file_path
