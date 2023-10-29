import threading


class ThreadSafeDict:
    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def get(self, key):
        # with self._lock:
        return self._data.get(key)

    def remove(self, key):
        with self._lock:
            if key in self._data:
                del self._data[key]

    def is_exists(self, key):
        if key in self._data:
            return True
        return False

    def get_all_keys(self):
        with self._lock:
            return list(self._data.keys())

    def clear(self):
        with self._lock:
            self._data = {}
