import torch
from torch import Tensor
from torch.utils.data import Sampler
from typing import Iterator, Sequence


class SubsetPoissonSampler(Sampler[int]):
    
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], sample_rate:float, generator=None) -> None:
        self.indices = indices
        self.num_samples = len(indices)
        self.sample_rate = sample_rate
        self.generator = generator

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    def __iter__(self) -> Iterator[int]:
        mask = (
            torch.rand(self.num_samples, generator=self.generator)
            < self.sample_rate
        )
        idx = mask.nonzero(as_tuple=False).reshape(-1).tolist()
        for i in idx:
            yield self.indices[i]

    def __len__(self) -> int:
        return self.num_samples
    
