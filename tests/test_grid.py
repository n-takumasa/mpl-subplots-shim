from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from mpl_subplots_shim._grid import Grid1D, Grid2D
from typing_extensions import assert_type

nrows = 2
ncols = 3


@pytest.fixture
def ax() -> Grid2D[Axes]:
    plt.switch_backend("agg")
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    return cast(Grid2D[Axes], cast(object, ax))


def test_grid2d(ax: Grid2D[Axes]):
    _ = ax[()]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[0]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0,]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0, 0]
    assert_type(_, Axes)
    assert isinstance(_, Axes)

    _ = ax[:]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[:,]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[:, :]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[0, :]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[:, 0]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (nrows,)

    _ = ax[...]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[...,]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[..., :]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[:, ...]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _ = ax[..., 0]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (nrows,)
    _ = ax[0, ...]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)

    _ = ax[[True] * nrows]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)

    _ = ax[[]]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (0, ncols)
    _ = ax[[0]]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)
    _ = ax[(0,),]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)
    _ = ax[[0],]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)

    _ = ax[(0,), (0,)]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (1,)
    _ = ax[(0, 0), (0, 0)]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (2,)
    _ = ax[[0], [0]]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (1,)
    _ = ax[[0, 0], [0, 0]]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (2,)

    _ = ax[(0,), :]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)
    _ = ax[(0,), ...]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)
    _ = ax[[0], :]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)
    _ = ax[[0], ...]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)

    _ = ax[:, (0,)]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, 1)
    _ = ax[..., (0,)]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, 1)
    _ = ax[:, [0]]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, 1)
    _ = ax[..., [0]]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, 1)


def test_numpy(ax: Grid2D[Axes]):
    _idx = np.ones(nrows, dtype=np.bool_)
    _ = ax[_idx]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (nrows, ncols)
    _idx = np.array([0], dtype=np.intp)
    _ = ax[_idx]
    assert_type(_, Grid2D[Axes])
    assert _.shape == (1, ncols)

    _idx = np.full((nrows, ncols), True, dtype=np.bool_)
    _ = ax[_idx]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (nrows * ncols,)
    _idx = [[True] * ncols] * nrows
    _ = ax[_idx]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (nrows * ncols,)


def test_grid1d(ax: Grid2D[Axes]):
    _ = ax[0][0]
    assert_type(_, Axes)
    assert isinstance(_, Axes)
    _ = ax[0][:]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0][...]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0][()]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0][0,]
    assert_type(_, Axes)
    assert isinstance(_, Axes)

    _ = ax[0][:,]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (ncols,)
    _ = ax[0][[],]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (0,)
    _ = ax[0][(),]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (0,)
    _ = ax[0][[0]]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (1,)
    _ = ax[0][(0,),]
    assert_type(_, Grid1D[Axes])
    assert _.shape == (1,)
