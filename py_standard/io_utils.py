import os
import glob
import shutil
import logging
from logging.handlers import TimedRotatingFileHandler


def use_logger():
    # 設置日誌設定
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # 設置日誌處理器
    log_file = 'logs/mylog.log'  # 日誌檔案路徑和檔名
    handler = TimedRotatingFileHandler(log_file, when='D', interval=1, backupCount=2)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 添加日誌處理器到根日誌器
    logger = logging.getLogger('')
    logger.addHandler(handler)
    return logger


def split_file_path(file_path: str):
    directory = os.path.dirname(file_path)
    full_filename = get_full_filename(file_path)
    filename, extension = split_filename(full_filename)
    return directory, filename, extension


def split_filename(full_filename: str):
    ss = os.path.splitext(full_filename)
    return ss[0], ss[1]


def get_full_filename(file_path: str):
    basename = os.path.basename(file_path)
    filename, file_ext = split_filename(basename)
    return filename + file_ext


def read_all_lines_file(file_path: str):
    encodings = ['utf-8-sig', 'utf-8', 'big5', 'ascii']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
            return lines
        except UnicodeDecodeError:
            continue
    else:
        raise Exception(f"Failed to open file {file_path} with any encoding.")


def query_files(folder_path: str, extensions: list[str]):
    for extension in extensions:
        pattern = os.path.join(folder_path, f'*{extension}')
        for file in glob.glob(pattern):
            yield file


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


def query_folders(folder_path: str):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            yield item_path


def make_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_target_file_path(source_file_path: str, target_dir: str):
    filename = os.path.basename(source_file_path)
    destination = os.path.join(target_dir, filename)
    return destination


def move_file(source_file_path: str, target_dir: str):
    # directory = os.path.dirname(source_file_path)
    filename = os.path.basename(source_file_path)
    destination = os.path.join(target_dir, filename)
    make_dir(target_dir)
    shutil.move(source_file_path, destination)


def copy_file(file_path: str, target_dir: str):
    filename = os.path.basename(file_path)
    destination = os.path.join(target_dir, filename)
    make_dir(target_dir)
    shutil.copy(file_path, destination)