import random
import torch
from torch.utils.data.sampler import Sampler
from utils.task_constructor import TaskConstructor

class TrainBatchSampler(Sampler):
    # task is of class TaskConstructor
    def __init__(self, params, task: TaskConstructor):
        self.batch_count = params["batch_count"]
        self.batch_size = params["batch_size"]
        self.task = task
        self.rng = random.Random(params["seed"])

        self.params = params
        self.params["rng"] = self.rng

    def __iter__(self):
        for _ in range(len(self)):
            yield self.sample()

    def __len__(self):
        return self.batch_count

    def sample(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(self.task.sample(self.params))
        return batch


class EvalBatchSampler(Sampler):
    def __init__(self, params, task: TaskConstructor):
        self.leave_last = params["leave_last"]
        self.batch_size = params["batch_size"]
        self.task = task
        self.mask_len = task.mask_len
        self.rng = random.Random(params["seed"])

        self.params = params
        self.params["rng"] = self.rng

    def __iter__(self):
        for batch_num in range(len(self)):
            yield self.sample(batch_num)

    def __len__(self):
        if self.leave_last:
            return self.mask_len // self.batch_size
        else:
            return (self.mask_len + self.batch_size - 1) // self.batch_size

    def sample(self, batch_number):
        batch = []
        start = batch_number * self.batch_size
        for i in range(self.batch_size):
            if start + i < self.mask_len:
                batch.append(self.task.sample(self.params, eval_idx=start + i))
        return batch