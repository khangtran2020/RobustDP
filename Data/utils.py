import torch
from torch import Tensor
from torch.utils.data import Sampler
from typing import Iterator, Sequence


class SubsetPoissonSampler(Sampler[int]):
    
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], sample_rate:float, steps:int=None, generator=None) -> None:
        self.indices = torch.LongTensor(indices)
        self.num_samples = len(indices)
        self.sample_rate = sample_rate
        self.generator = generator

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

        if steps is not None:
            self.steps = steps
        else:
            self.steps = int(1 / self.sample_rate)

    def __iter__(self) -> Iterator[int]:

        num_batches = self.steps
        while num_batches > 0:
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
            )
            idx = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            yield self.indices[idx].tolist()
            num_batches -= 1

    def __len__(self):
        return self.steps
    
