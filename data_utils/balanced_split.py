import bisect
import warnings
import math
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union
)

# No 'default_generator' in torch/__init__.pyi
from torch import default_generator, randperm, Generator, Tensor
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset, Dataset

"""
__all__ = [
    "Dataset",
    "IterableDataset",
    "TensorDataset",
    "ConcatDataset",
    "ChainDataset",
    "Subset",
    "random_split",
    "balanced_split",
]
"""

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


def balanced_split(dataset: Dataset[T], split: float = 0.8) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """


    #indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    #indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]

    # Crea gli strati del dataset sulla base delle etichette delle classi
    stratified_df = dataset.info.groupby('source_type', group_keys=True).apply(lambda x: x.sample(frac=1, random_state=42))
    

    # Calcola la dimensione del dataset di training e di test in base alla percentuale di split desiderata
    #train_size = int(len(dataset) * split)
    #test_size = len(stratified_df) - train_size

    #stratified_df = stratified_df.reset_index(drop=True)
    # Seleziona casualmente gli elementi di ogni strato per creare il dataset di training e di test
    #train_df = stratified_df.groupby('source_type', group_keys=True).apply(lambda x: x.sample(n=train_size, random_state=42))
    train_df = stratified_df.sample(frac=split)
    train_df = train_df.reset_index(level="source_type", drop=True).sort_index()
    test_df = dataset.info.drop(train_df.index)
    test_df = test_df.sort_index()

    return [Subset(dataset, train_df.index), Subset(dataset, test_df.index)]


#change to proportioned term
def quotas_balanced_split(dataset: Dataset[T], split: float = 0.8, max_per_class_factor = 3) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    #indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    #indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]

    # Crea gli strati del dataset sulla base delle etichette delle classi
    #stratified_df = dataset.info.groupby('source_type', group_keys=True).apply(lambda x: x.sample(frac=1, random_state=42))
    min = 99999999999
    grouped = dataset.info.groupby('source_type', group_keys=True)
    train_set = grouped.sample(frac=split)
    grouped = train_set.groupby('source_type', group_keys=True)
    #dataset.info

    for name, group in grouped:
        #print(name)
        #print(group.count()[0])
        if group.count()[0] < min:
            min = group.count()[0]

    limit = max_per_class_factor * min
    for name, group in grouped:
        #print(name)
        #print(group.count()[0])
        #print(train_set.count()[0])
        if group.count()[0] > limit:
            train_set = train_set.drop(group.index[limit:])
            #print(train_set.count()[0])


    # Calcola la dimensione del dataset di training e di test in base alla percentuale di split desiderata
    #train_size = int(len(dataset) * split)
    #test_size = len(stratified_df) - train_size

    #stratified_df = stratified_df.reset_index(drop=True)
    # Seleziona casualmente gli elementi di ogni strato per creare il dataset di training e di test
    #train_df = stratified_df.groupby('source_type', group_keys=True).apply(lambda x: x.sample(n=train_size, random_state=42))
    #train_set = train_set.reset_index(level="source_type", drop=True).sort_index()
    train_set = train_set.sort_index()
    test_df = dataset.info.drop(train_set.index)
    test_df = test_df.sort_index()

    return [Subset(dataset, train_set.index), Subset(dataset, test_df.index)]