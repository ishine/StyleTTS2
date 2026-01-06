from collections.abc import Callable, Sequence
from functools import reduce
from inspect import isfunction
from math import ceil, floor, log2
from typing import TypeGuard, TypeVar

import torch

T = TypeVar("T")


def exists(val: T | None) -> TypeGuard[T]:
    return val is not None


def iff(condition: bool, value: T) -> T | None:
    return value if condition else None


def is_sequence(obj: T) -> TypeGuard[list | tuple]:
    return isinstance(obj, (list, tuple))


def default(val: T | None, d: Callable[..., T] | T) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def to_list(val: T | Sequence[T]) -> list[T]:
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return [val]  # type: ignore


def prod(vals: Sequence[int]) -> int:
    return reduce(lambda x, y: x * y, vals)


def closest_power_2(x: float) -> int:
    exponent = log2(x)
    distance_fn = lambda z: abs(x - 2**z)  # noqa
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)


def rand_bool(shape, proba, device=None):
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


"""
Kwargs Utils
"""


def group_dict_by_prefix(prefix: str, d: dict) -> tuple[dict, dict]:
    return_dicts: tuple[dict, dict] = ({}, {})
    for key in d:
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


def groupby(prefix: str, d: dict, keep_prefix: bool = False) -> tuple[dict, dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs


def prefix_dict(prefix: str, d: dict) -> dict:
    return {prefix + str(k): v for k, v in d.items()}
