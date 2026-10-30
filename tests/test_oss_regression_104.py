import types
import typing

import pytest

from typing_inspect import get_origin


UNION_CASES = [
    (typing.Union[int, str], typing.Union),
    (typing.Optional[int], typing.Union),
]
if hasattr(types, "UnionType"):
    UNION_CASES.extend([
        (int | str, types.UnionType),
        (int | None, types.UnionType),
        ((int | str) | float, types.UnionType),
        (int | list[str], types.UnionType),
    ])


@pytest.mark.parametrize("union, expected", UNION_CASES)
def test_union_origin_matches_standard_library(union, expected):
    origin = None
    error = None
    try:
        origin = get_origin(union)
    except AttributeError as exc:
        error = exc
    assert error is None, "get_origin({!r}) raised {!r}".format(union, error)
    assert origin is expected
    if hasattr(typing, "get_origin"):
        assert origin is typing.get_origin(union)
