import os


def get_data_file_path(file_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ml/data/{file_name}")
