from typing import Iterable


def min_max(iterable: Iterable[int | float]):
    min = max = None
    for value in iterable:
        if min is None or value < min:
            min = value
        if max is None or value > max:
            max = value
    return min, max
