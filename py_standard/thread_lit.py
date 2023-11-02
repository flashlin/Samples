import asyncio

from logger_lit import build_logger


class ThreadWorkerBase:
    def __init__(
        self,
        worker_id: str,
        limit_worker_concurrency: int = 1,
    ):
        self.worker_id = worker_id
        self.limit_worker_concurrency = limit_worker_concurrency
        self.semaphore = None
        self.logger = build_logger("thread_worker", f"thread_worker_{self.worker_id}.log")

    def acquire_semaphore(self):
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.limit_worker_concurrency)
        return self.semaphore.acquire()

    def release_semaphore(self):
        self.semaphore.release()

    def process(self, params):
        raise NotImplementedError

    def get_queue_length(self):
        if (
            self.semaphore is None
            or self.semaphore._value is None
            or self.semaphore._waiters is None
        ):
            return 0
        else:
            return (
                self.limit_worker_concurrency
                - self.semaphore._value
                + len(self.semaphore._waiters)
            )

    def get_status(self):
        return {
            "queue_length": self.get_queue_length(),
        }
