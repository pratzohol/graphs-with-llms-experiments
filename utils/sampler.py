import random
from torch.utils.data.sampler import Sampler
from utils.task_constructor import TaskConstructor

class BatchSampler(Sampler):
    # task is of class TaskConstructor
    def __init__(self, params, task: TaskConstructor):
        self.batch_count = params["batch_count"]
        self.batch_size = params["batch_size"]
        self.task = task
        self.rng = random.Random(params["seed"])

        self.params = params
        self.params["rng"] = self.rng

    def __iter__(self):
        for _ in range(self.batch_count):
            yield self.sample()

    def __len__(self):
        return self.batch_count

    def sample(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(self.task.sample(**self.params))
        return batch
