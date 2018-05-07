from typing_inspect import (
    is_generic_type, is_callable_type, is_tuple_type, is_union_type,
    is_typevar, is_classvar, get_origin, get_parameters, get_last_args, get_args,
    get_generic_type, get_generic_bases, get_last_origin, typed_dict_keys,
)
from unittest import TestCase, main, skipIf, skipUnless
from typing import (
    Union, ClassVar, Callable, Optional, TypeVar, Sequence, Mapping,
    MutableMapping, Iterable, Generic, List, Any, Dict, Tuple, NamedTuple,
)

import sys
NEW_TYPING = sys.version_info[:3] >= (3, 7, 0)  # PEP 560

from mypy_extensions import TypedDict

PY36_TESTS = """
class TD(TypedDict):
    x: int
    y: int
class Other(dict):
    x: int
    y: int
"""

PY36 = sys.version_info[:3] >= (3, 6, 0)
if PY36:
    exec(PY36_TESTS)


class IsUtilityTestCase(TestCase):
    def sample_test(self, fun, samples, nonsamples):
        for s in samples:
            self.assertTrue(fun(s))
        for s in nonsamples:
            self.assertFalse(fun(s))

    def test_generic(self):
        T = TypeVar('T')
        samples = [Generic, Generic[T], Iterable[int], Mapping,
                   MutableMapping[T, List[int]], Sequence[Union[str, bytes]]]
        nonsamples = [int, Union[int, str], Union[int, T], ClassVar[List[int]],
                      Callable[..., T], ClassVar, Optional, bytes, list]
        self.sample_test(is_generic_type, samples, nonsamples)

    def test_callable(self):
        samples = [Callable, Callable[..., int],
                   Callable[[int, int], Iterable[str]]]
        nonsamples = [int, type, 42, [], List[int],
                      Union[callable, Callable[..., int]]]
        self.sample_test(is_callable_type, samples, nonsamples)
        class MyClass(Callable[[int], int]):
            pass
        self.assertTrue(is_callable_type(MyClass))

    def test_tuple(self):
        samples = [Tuple, Tuple[str, int], Tuple[Iterable, ...]]
        nonsamples = [int, tuple, 42, List[int], NamedTuple('N', [('x', int)])]
        self.sample_test(is_tuple_type, samples, nonsamples)
        class MyClass(Tuple[str, int]):
            pass
        self.assertTrue(is_tuple_type(MyClass))

    def test_union(self):
        T = TypeVar('T')
        S = TypeVar('S')
        samples = [Union, Union[T, int], Union[int, Union[T, S]]]
        nonsamples = [int, Union[int, int], [], Iterable[Any]]
        self.sample_test(is_union_type, samples, nonsamples)

    def test_typevar(self):
        T = TypeVar('T')
        S_co = TypeVar('S_co', covariant=True)
        samples = [T, S_co]
        nonsamples = [int, Union[T, int], Union[T, S_co], type, ClassVar[int]]
        self.sample_test(is_typevar, samples, nonsamples)

    def test_classvar(self):
        T = TypeVar('T')
        samples = [ClassVar, ClassVar[int], ClassVar[List[T]]]
        nonsamples = [int, 42, Iterable, List[int], type, T]
        self.sample_test(is_classvar, samples, nonsamples)


class GetUtilityTestCase(TestCase):

    @skipIf(NEW_TYPING, "Not supported in Python 3.7")
    def test_last_origin(self):
        T = TypeVar('T')
        self.assertEqual(get_last_origin(int), None)
        self.assertEqual(get_last_origin(ClassVar[int]), None)
        self.assertEqual(get_last_origin(Generic[T]), Generic)
        self.assertEqual(get_last_origin(Union[T, int][str]), Union[T, int])
        self.assertEqual(get_last_origin(List[Tuple[T, T]][int]), List[Tuple[T, T]])
        self.assertEqual(get_last_origin(List), List)

    def test_origin(self):
        T = TypeVar('T')
        self.assertEqual(get_origin(int), None)
        self.assertEqual(get_origin(ClassVar[int]), None)
        self.assertEqual(get_origin(Generic), Generic)
        self.assertEqual(get_origin(Generic[T]), Generic)
        self.assertEqual(get_origin(List[Tuple[T, T]][int]), list if NEW_TYPING else List)

    def test_parameters(self):
        T = TypeVar('T')
        S_co = TypeVar('S_co', covariant=True)
        U = TypeVar('U')
        self.assertEqual(get_parameters(int), ())
        self.assertEqual(get_parameters(Generic), ())
        self.assertEqual(get_parameters(Union), ())
        self.assertEqual(get_parameters(List[int]), ())
        self.assertEqual(get_parameters(Generic[T]), (T,))
        self.assertEqual(get_parameters(Tuple[List[T], List[S_co]]), (T, S_co))
        self.assertEqual(get_parameters(Union[S_co, Tuple[T, T]][int, U]), (U,))
        self.assertEqual(get_parameters(Mapping[T, Tuple[S_co, T]]), (T, S_co))

    @skipIf(NEW_TYPING, "Not supported in Python 3.7")
    def test_last_args(self):
        T = TypeVar('T')
        S = TypeVar('S')
        self.assertEqual(get_last_args(int), ())
        self.assertEqual(get_last_args(Union), ())
        self.assertEqual(get_last_args(ClassVar[int]), (int,))
        self.assertEqual(get_last_args(Union[T, int]), (T, int))
        self.assertEqual(get_last_args(Iterable[Tuple[T, S]][int, T]), (int, T))
        self.assertEqual(get_last_args(Callable[[T, S], int]), (T, S, int))
        self.assertEqual(get_last_args(Callable[[], int]), (int,))

    @skipIf(NEW_TYPING, "Not supported in Python 3.7")
    def test_args(self):
        T = TypeVar('T')
        self.assertEqual(get_args(Union[int, Tuple[T, int]][str]),
                         (int, (Tuple, str, int)))
        self.assertEqual(get_args(Union[int, Union[T, int], str][int]),
                         (int, str))
        self.assertEqual(get_args(int), ())

    def test_args_evaluated(self):
        T = TypeVar('T')
        self.assertEqual(get_args(Union[int, Tuple[T, int]][str], evaluate=True),
                         (int, Tuple[str, int]))
        self.assertEqual(get_args(Dict[int, Tuple[T, T]][Optional[int]], evaluate=True),
                         (int, Tuple[Optional[int], Optional[int]]))
        self.assertEqual(get_args(Callable[[], T][int], evaluate=True), ([], int,))

    def test_generic_type(self):
        T = TypeVar('T')
        class Node(Generic[T]): pass
        self.assertIs(get_generic_type(Node()), Node)
        self.assertIs(get_generic_type(Node[int]()), Node[int])
        self.assertIs(get_generic_type(Node[T]()), Node[T],)
        self.assertIs(get_generic_type(1), int)

    def test_generic_bases(self):
        class MyClass(List[int], Mapping[str, List[int]]): pass
        self.assertEqual(get_generic_bases(MyClass),
                         (List[int], Mapping[str, List[int]]))
        self.assertEqual(get_generic_bases(int), ())

    @skipUnless(PY36, "Python 3.6 required")
    def test_typed_dict(self):
        TDOld = TypedDict("TDOld", {'x': int, 'y': int})
        self.assertEqual(typed_dict_keys(TD), {'x': int, 'y': int})
        self.assertEqual(typed_dict_keys(TDOld), {'x': int, 'y': int})
        self.assertIs(typed_dict_keys(dict), None)
        self.assertIs(typed_dict_keys(Other), None)
        self.assertIsNot(typed_dict_keys(TD), TD.__annotations__)


if __name__ == '__main__':
    main()
