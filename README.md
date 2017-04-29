Typing Inspect
==============

The ``typing_inspect`` module defines experimental API for runtime
inspection of types defined in the Python standard ``typing`` module.
Works with ``typing`` version ``3.5.3`` and later. Example usage:

```python
from typing import Generic, TypeVar, Iterable, Mapping, Union
from typing_inspect import is_generic_type

T = TypeVar('T')

class MyCollection(Generic[T]):
    content: T

assert is_generic_type(Mapping)
assert is_generic_type(Iterable[int])
assert is_generic_type(MyCollection[T])

assert not is_generic_type(int)
assert not is_generic_type(Union[int, T])
```

Currently provided functions:
* ``is_generic_type``:
  Test if the given type is a generic type. This includes ``Generic`` itself,
  but excludes special typing constructs such as ``Union``, ``Tuple``,
  ``Callable``, ``ClassVar``.
* ``is_callable_type``:
  Test if the type is a generic callable type, including subclasses
  excluding non-generic types and callables.
* ``is_tuple_type``:
  Test if the type is a generic tuple type, including subclasses excluding
  non-generic classes.
* ``is_union_type``:
  Test if the type is a union type.
* ``is_typevar``:
  Test if the type represents a type variable.
* ``is_classvar``:
  Test if the type represents a class variable.
* ``get_origin``:
  Get the unsubscripted version of a type. Supports generic types, ``Union``,
  ``Callable``, and ``Tuple``. Returns ``None`` for unsupported types.
* ``get_last_origin``:
  Get the last base of (multiply) subscripted type. Supports generic types,
  ``Union``, ``Callable``, and ``Tuple``. Returns ``None`` for unsupported
  types.
* ``get_parameters``:
  Return type parameters of a parameterizable type as a tuple
  in lexicographic order. Parameterizable types are generic types,
  unions, tuple types and callable types.
* ``get_args``:
  Get type arguments with all substitutions performed. For unions,
  basic simplifications used by ``Union`` constructor are performed.
  If ``evaluate`` is ``False`` (default), report result as nested tuple,
  this matches the internal representation of types. If ``evaluate`` is
  ``True``, then all type parameters are applied (this could be time and
  memory expensive).
* ``get_last_args``:
  Get last arguments of (multiply) subscripted type.
  Parameters for ``Callable`` are flattened.
* ``get_generic_type``:
  Get the generic type of an object if possible, or runtime class otherwise.
* ``get_generic_bases``:
  Get generic base types of a type or empty tuple if not possible.
