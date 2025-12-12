from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import EllipsisType
from typing import Generic, Literal, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from typing_extensions import (
    Any,
    Protocol,
    Self,
    SupportsIndex,
    overload,
)

_Idx: TypeAlias = int | np.integer
_Seq: TypeAlias = (
    onp.Array1D[np.bool] | onp.Array1D[np.integer] | list[bool] | list[int]
)
_Rest: TypeAlias = slice | EllipsisType
_Adv: TypeAlias = tuple[int, ...] | _Seq

ScalarType = TypeVar("ScalarType")


class FlatIter(Generic[ScalarType], Protocol):
    """np.flatiter[~ScalarType]"""

    @property
    def base(self) -> npt.NDArray[Any]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> ScalarType: ...
    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, key: _Idx | tuple[_Idx]) -> ScalarType: ...
    @overload
    def __getitem__(
        self, key: _Seq | _Rest | tuple[_Seq | _Rest | _Adv]
    ) -> Grid1D[ScalarType]: ...


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
        self, key: tuple[()] | _Seq | _Rest | tuple[_Seq | _Rest | _Adv]
    ) -> Self: ...
    @overload
    def __getitem__(self, key: _Idx | tuple[_Idx]) -> ScalarType: ...


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
        | _Seq
        | _Rest
        | tuple[_Rest | _Adv]
        | tuple[_Rest, _Rest]
        | tuple[_Adv, _Rest]
        | tuple[_Rest, _Adv],
    ) -> Self: ...

    @overload
    def __getitem__(
        self,
        key: _Idx
        | tuple[_Idx]
        | tuple[_Idx, _Rest]
        | tuple[_Rest, _Idx]
        | tuple[_Adv, _Adv]
        | onp.Array2D[np.bool]
        | list[list[bool]],
    ) -> Grid1D[ScalarType]: ...

    @overload
    def __getitem__(self, key: tuple[_Idx, _Idx]) -> ScalarType: ...
