import torch
from torch import Tensor
from torch.utils.data import Sampler
from torch.utils.data.dataloader import _collate_fn_t
from typing import Iterator, Sequence, Optional, Union, Tuple, Type, Any

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
    
def wrap_collate_with_empty(
    *,
    collate_fn: Optional[_collate_fn_t],
    sample_empty_shapes: Sequence[Tuple],
    dtypes: Sequence[Union[torch.dtype, Type]],
):

    def collate(batch):
        if len(batch) > 0:
            return collate_fn(batch)
        else:
            return [
                torch.zeros(shape, dtype=dtype)
                for shape, dtype in zip(sample_empty_shapes, dtypes)
            ]

    return collate

def shape_safe(x: Any) -> Tuple:
    """
    Exception-safe getter for ``shape`` attribute

    Args:
        x: any object

    Returns:
        ``x.shape`` if attribute exists, empty tuple otherwise
    """
    return x.shape if hasattr(x, "shape") else ()

def dtype_safe(x: Any) -> Union[torch.dtype, Type]:
    """
    Exception-safe getter for ``dtype`` attribute

    Args:
        x: any object

    Returns:
        ``x.dtype`` if attribute exists, type of x otherwise
    """
    return x.dtype if hasattr(x, "dtype") else type(x)
