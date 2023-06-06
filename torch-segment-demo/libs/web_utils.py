
import os

static_folder = './public'

def get_html(relative_file_path: str):
    """return rendered html"""
    with open(os.path.join(static_folder, relative_file_path)) as f:
        return f.read()


