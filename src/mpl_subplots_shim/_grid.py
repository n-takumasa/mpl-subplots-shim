from __future__ import annotations

from collections.abc import Collection, Iterator, Reversible, Sequence
from types import EllipsisType
from typing import ClassVar, Generic, Literal, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from typing_extensions import Any, Protocol, Self, SupportsIndex, overload

ScalarType = TypeVar("ScalarType")

_T_co = TypeVar("_T_co", covariant=True)


class SequenceNotTuple(Reversible[_T_co], Collection[_T_co], Protocol[_T_co]):
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _T_co: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...
    def index(self, value: Any, start: int = 0, stop: int = ..., /) -> int: ...
    def count(self, value: Any, /) -> int: ...

    __hash__: ClassVar[None]  # type: ignore  # ty: ignore[unused-ignore-comment]


# TODO: 2D bool mask

# Advanced indexing is triggered when the selection object, obj,
_Adv: TypeAlias = (
    # is a non-tuple sequence object,
    SequenceNotTuple[SupportsIndex]
    | SequenceNotTuple[bool]
    # an ndarray (of data type integer or bool),
    # | onp.Array1D[np.integer]
    # | onp.Array1D[np.bool_]
    | np.ndarray[tuple[int,], np.dtype[np.integer]]
    | np.ndarray[tuple[int,], np.dtype[np.bool_]]
    # | np.ndarray[Any, np.dtype[np.integer]]
    # | np.ndarray[Any, np.dtype[np.bool_]]
)
# or a tuple with at least one sequence object or ndarray
# (of data type integer or bool).
# There are two types of advanced indexing: integer and Boolean.
_AdvInTuple: TypeAlias = _Adv | Sequence[SupportsIndex] | Sequence[bool]


class FlatIter(Generic[ScalarType], Protocol):
    """np.flatiter[~ScalarType]"""

    @property
    def base(self) -> npt.NDArray[Any]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> ScalarType: ...
    def __len__(self) -> int: ...

    @overload
    def __getitem__(
        self,
        key: tuple[()]
        | _Adv
        | slice
        | EllipsisType
        | tuple[slice | EllipsisType,]
        | tuple[_AdvInTuple,],
    ) -> Grid1D[ScalarType]: ...
    @overload
    def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex,]) -> ScalarType: ...


class Grid1D(Generic[ScalarType], Protocol):
    """np.ndarray[tuple[int], ~ScalarType]"""

    @property
    def base(self) -> npt.NDArray[Any] | None: ...
    @property
    def ndim(self) -> Literal[1]: ...
    @property
    def size(self) -> int: ...
    @property
    def shape(self) -> tuple[int,]: ...
    @property
    def flat(self) -> FlatIter[ScalarType]: ...

    def item(self, i0: SupportsIndex | tuple[SupportsIndex], /) -> ScalarType: ...

    def flatten(self, /, order: onp.OrderKACF = "C") -> Self: ...
    def ravel(self, /, order: onp.OrderKACF = "C") -> Self: ...
    def tolist(self) -> list[ScalarType]: ...

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[ScalarType]: ...

    @property
    def T(self) -> Self: ...
    @overload
    def transpose(self, axes: SupportsIndex | Sequence[SupportsIndex]) -> Self: ...
    @overload
    def transpose(self, *axes: SupportsIndex) -> Self: ...

    @overload
    def __getitem__(
        self,
        key: tuple[()]
        | _Adv
        | slice
        | EllipsisType
        | tuple[slice | EllipsisType,]
        | tuple[_AdvInTuple,],
    ) -> Self: ...
    @overload
    def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex,]) -> ScalarType: ...


class Grid2D(Generic[ScalarType], Protocol):
    """np.ndarray[tuple[int, int], ~ScalarType]"""

    @property
    def base(self) -> npt.NDArray[Any] | None: ...
    @property
    def ndim(self) -> Literal[2]: ...
    @property
    def size(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def flat(self) -> FlatIter[ScalarType]: ...

    @overload
    def item(
        self, i0: SupportsIndex | tuple[SupportsIndex, SupportsIndex]
    ) -> ScalarType: ...
    @overload
    def item(self, i0: SupportsIndex, i1: SupportsIndex, /) -> ScalarType: ...

    def flatten(self, /, order: onp.OrderKACF = "C") -> Grid1D[ScalarType]: ...
    def ravel(self, /, order: onp.OrderKACF = "C") -> Grid1D[ScalarType]: ...
    def tolist(self) -> list[list[ScalarType]]: ...

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Grid1D[ScalarType]]: ...

    @property
    def T(self) -> Self: ...
    @overload
    def transpose(self, axes: SupportsIndex | Sequence[SupportsIndex]) -> Self: ...
    @overload
    def transpose(self, *axes: SupportsIndex) -> Self: ...

    @overload
    def __getitem__(
        self,
        key: tuple[()]
        | slice
        | tuple[slice,]
        | tuple[slice, slice]
        | EllipsisType
        | tuple[EllipsisType,]
        | tuple[EllipsisType, slice]
        | tuple[slice, EllipsisType]
        | _Adv
        | tuple[_AdvInTuple,]
        | tuple[_AdvInTuple, slice | EllipsisType]
        | tuple[slice | EllipsisType, _AdvInTuple],
    ) -> Self: ...

    @overload
    def __getitem__(
        self,
        key: SupportsIndex
        | tuple[SupportsIndex,]
        | tuple[SupportsIndex, slice | EllipsisType]
        | tuple[slice | EllipsisType, SupportsIndex]
        | tuple[_AdvInTuple, _AdvInTuple]
        | tuple[_AdvInTuple, SupportsIndex]
        | tuple[SupportsIndex, _AdvInTuple],
    ) -> Grid1D[ScalarType]: ...

    # boolean index
    @overload
    def __getitem__(
        self,
        key: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
        | Sequence[Sequence[bool]]
        | tuple[np.ndarray[tuple[int, int], np.dtype[np.bool_]],]
        | tuple[Sequence[Sequence[bool]],],
    ) -> Grid1D[ScalarType]: ...

    # boolean index
    @overload
    def __getitem__(
        self,
        key: np.ndarray[Any, np.dtype[np.bool_]]
        | tuple[np.ndarray[Any, np.dtype[np.bool_]],],
    ) -> Self | Grid1D[ScalarType]: ...

    @overload
    def __getitem__(
        self,
        key: tuple[SupportsIndex, SupportsIndex],
    ) -> ScalarType: ...
