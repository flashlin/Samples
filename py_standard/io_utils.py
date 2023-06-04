import os
import glob
import shutil


def query_sub_files(folder: str, ext_list: list[str]):
    """
    :param folder:
    :param ext_list: ['.png', '.jpg']
    :return:
    """
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(tuple(ext_list)):
                file_path = os.path.join(root, file)
                yield file_path


def make_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def move_file(source_file_path: str, target_dir: str):
    # directory = os.path.dirname(source_file_path)
    filename = os.path.basename(source_file_path)
    destination = os.path.join(target_dir, filename)
    make_dir(destination)
    shutil.move(source_file_path, destination)
