from celery import Celery, signals
from model_utils import create_llama2, generate_output2


def make_celery(app_name=__name__):
    backend = broker = 'redis://gpt_redis_1:6379/0'
    return Celery(app_name, backend=backend, broker=broker)


celery = make_celery()

llm_model = None


@signals.worker_process_init.connect
def setup_model(signal, sender, **kwargs):
    global llm_model
    llm_model = create_llama2()


@celery.task
def generate_text_task(prompt):
    time, memory, outputs = generate_output2(
        prompt, llm_model
    )
    return outputs, time, memory
