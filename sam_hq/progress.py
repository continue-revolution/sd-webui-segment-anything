
import time
from modules.priority_lock import PriorityLock
queue_lock = PriorityLock("Segment")

class QueueLock:
    def __init__(self, pri=100, name=None):
        self._priority = pri
        self._name = name

    def __enter__(self):
        queue_lock.acquire(self._priority, self._name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        queue_lock.release()


current_task = None
pending_tasks = {}
finished_tasks = []
recorded_results = []
recorded_results_limit = 5
task_results = {}

def save_task_result(task_id, result):
    task_results[task_id] = result
    if len(task_results) > 5:
        task_results.pop(list(task_results.keys())[0])

def get_task_result(task_id):
    if task_id in task_results:
        return task_results[task_id]
    return None

def start_task(id_task):
    global current_task
    current_task = id_task
    pending_tasks.pop(id_task, None)


def finish_task(id_task):
    global current_task
    if current_task == id_task:
        current_task = None

    finished_tasks.append(id_task)
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)


def record_results(id_task, res):
    recorded_results.append((id_task, res))
    if len(recorded_results) > recorded_results_limit:
        recorded_results.pop(0)


def get_task_info(task_id):
    active = task_id == current_task
    queued = task_id in pending_tasks
    completed = task_id in finished_tasks
    pos, total = None, None
    if not active:
        pos, total = queue_lock.get_task_position(task_id)
    ret = {}
    ret['active'] = active
    ret['queued'] = queued
    ret['queue_pos'] = pos
    ret['queue_len'] = total
    ret['completed'] = completed
    return ret


def add_task_to_queue(id_job):
    pending_tasks[id_job] = time.time()

