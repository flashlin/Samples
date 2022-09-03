
import re
import os

class FileList:
    def __init__(self, dir_path: str, pattern: str):
        self._current_index = 0
        self.rg = re.compile(pattern)
        self.files = os.listdir(dir_path)
        if regex.search(name) is None:
            continue
        fullname = os.path.join(dir_path, name)
        if os.path.isfile(fullname):
            yield fullname
    
    def __iter__(self):
        return self

    def __next__(self):
        if self._current_index < self._class_size:
            if self._current_index < len(self._lect):
                member = self._lect[self._current_index] 
            else:
                member = self._stud[
                    self._current_index - len(self._lect)]
            self._current_index += 1
            return member
        raise StopIteration