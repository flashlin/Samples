from dataclasses import dataclass

import torch.multiprocessing as mp

request_queue = mp.SimpleQueue()
response_queue = mp.SimpleQueue()
lock = mp.Lock()
task_id = mp.Value('i', 0)

def increase_task_id():
    global lock
    new_task_id = 0
    with lock:
       task_id.value += 1
       new_task_id = task_id.value
    return new_task_id


@dataclass
class TaskInfo:
    id: int = 0
    args: object = None
    result: object = None

class QueueService:
    def __init__(self, work_fn):
        self.task_id = 0
        t = mp.Process(target=self.worker, args=(work_fn, ))
        t.start()

    @staticmethod
    def worker(work_fn):
        while True:
            if not request_queue.empty():
                task_info = request_queue.get()
                task_info.result = work_fn(task_info.args)
                response_queue.put(task_info)

    def queue(self, message):
        new_task_id = increase_task_id()
        request_queue.put(TaskInfo(
            id=new_task_id,
            args=message,
            result=None,
        ))
        return new_task_id

    def get_result(self, task_id: int):
        pass