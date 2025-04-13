import multiprocessing
import time
from multiprocessing import Pool, Queue

from tqdm import tqdm


# Worker function (must be at top-level for multiprocessing)
def worker(group_id, task_arg):
    try:
        time.sleep(0.5)  # Simulate work
        return (group_id, task_arg * 2)
    except Exception as e:
        return (group_id, f"Error: {e}")


class ProgressTracker:
    def __init__(self, result_queue):
        self.completed = 0
        self.total = 0
        self.group_counts = {}
        self.result_queue = result_queue
        self.pbar = tqdm(total=0, dynamic_ncols=True, desc="Progress")

    def update_progress(self):
        """Process results from the queue (call periodically from GUI thread)"""
        while not self.result_queue.empty():
            result = self.result_queue.get()
            if isinstance(result, Exception):
                print(f"Task failed: {result}")
                continue
            group_id, _ = result
            self.completed += 1
            self.group_counts[group_id] = self.group_counts.get(group_id, 0) + 1
            self.pbar.update(1)
            self.pbar.set_description(
                f"Completed: {self.completed}/{self.total} | Groups: {self.group_counts}"
            )

    def add_tasks(self, num_tasks):
        self.total += num_tasks
        self.pbar.total = self.total
        self.pbar.refresh()

    def close(self):
        self.pbar.close()


class TaskManager:
    def __init__(self):
        self.result_queue = Queue()
        self.pool = None
        self.tracker = ProgressTracker(self.result_queue)

    def start_pool(self):
        if self.pool is None:
            self.pool = Pool(processes=4)

    def submit_tasks(self, group_id, task_args):
        if self.pool is None:
            self.start_pool()

        for arg in task_args:
            self.pool.apply_async(
                worker,
                args=(group_id, arg),
                callback=lambda res: self.result_queue.put(res),
                error_callback=lambda e: self.result_queue.put(e)
            )
        self.tracker.add_tasks(len(task_args))

    def shutdown(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.tracker.close()


# Example usage pattern
if __name__ == '__main__':
    multiprocessing.freeze_support()

    manager = TaskManager()

    # Initial batch
    manager.submit_tasks('1', [1, 2, 3])
    manager.submit_tasks('1', [1, 2, 3])
    manager.submit_tasks('1', [1, 2, 3])
    manager.submit_tasks('1', [1, 2, 3])
    manager.submit_tasks('2', [1, 2, 3])
    manager.submit_tasks('2', [1, 2, 3])
    manager.submit_tasks('2', [1, 2, 3])
    manager.submit_tasks('3', [1, 2, 3])
    manager.submit_tasks('3', [1, 2, 3])
    manager.submit_tasks('2', [1, 2, 3])
    manager.submit_tasks('2', [1, 2, 3])
    manager.submit_tasks('1', [1, 2, 3])
    manager.submit_tasks('1', [1, 2, 3])

    # Simulate GUI main loop
    try:
        while True:
            time.sleep(0.1)  # Replace with GUI framework's event loop
            manager.tracker.update_progress()

            # Simulate adding more tasks later
            start_time = time.time()
            if time.time() > start_time + 1.5:  # Some condition
                manager.submit_tasks('group2', [6, 7, 8])
                start_time = time.time()

    except KeyboardInterrupt:
        manager.shutdown()
